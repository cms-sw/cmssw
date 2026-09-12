#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "L1Trigger/TrackerDTC/interface/Setup.h"
#include "L1Trigger/TrackerDTC/interface/SensorModule.h"
#include "L1Trigger/TrackerDTC/interface/StubFE.h"
#include "L1Trigger/TrackerDTC/interface/StubDTC.h"

#include <iterator>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <utility>

namespace trackerDTC {

  /*! \class  trackerDTC::ProducerDTC
   *  \brief  Class to produce hardware like structured TTStub Collection used by Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Jan
   */
  class ProducerDTC : public edm::stream::EDProducer<> {
  public:
    explicit ProducerDTC(const edm::ParameterSet&);
    ~ProducerDTC() override = default;

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;
    // apply cabling map, reorganise stub collections
    void consume(const edm::Handle<TTStubDetSetVec>&, std::vector<std::vector<std::vector<TTStubRef>>>&) const;
    // board level transforming and routing of stubs
    void produce(const std::vector<std::vector<TTStubRef>>&, TTDTC&) const;
    // ED input token of TTStubs
    edm::EDGetTokenT<TTStubDetSetVec> edGetToken_;
    // ED output token for accepted stubs
    edm::EDPutTokenT<TTDTC> edPutToken;
    // Setup token
    edm::ESGetToken<Setup, SetupRcd> esGetToken_;
    // helper class to store configurations
    const Setup* setup_ = nullptr;
    // this dtc
    int dtcId_;
  };

  ProducerDTC::ProducerDTC(const edm::ParameterSet& iConfig) {
    // book in- and output ED products
    const edm::InputTag& inputTag = iConfig.getParameter<edm::InputTag>("InputTag");
    const std::string& branch = iConfig.getParameter<std::string>("Branch");
    edGetToken_ = consumes(inputTag);
    edPutToken = produces(branch);
    // book ES products
    esGetToken_ = esConsumes();
  }

  void ProducerDTC::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetToken_);
    // empty DTC products
    TTDTC ttDCT(setup_->sysNumRegion(), setup_->sysNumOverlap(), setup_->regNumDTC());
    // read in stub collection
    edm::Handle<TTStubDetSetVec> handle;
    iEvent.getByToken(edGetToken_, handle);
    // apply cabling map, reorganise stub collections
    std::vector<std::vector<std::vector<TTStubRef>>> stubs(setup_->sysNumDTC(),
                                                           std::vector<std::vector<TTStubRef>>(setup_->dtcNumModule()));
    consume(handle, stubs);
    // board level processing
    for (dtcId_ = 0; dtcId_ < setup_->sysNumDTC(); dtcId_++)
      produce(stubs[dtcId_], ttDCT);
    // store ED products
    iEvent.emplace(edPutToken, std::move(ttDCT));
  }

  // apply cabling map, reorganise stub collections
  void ProducerDTC::consume(const edm::Handle<TTStubDetSetVec>& handle,
                            std::vector<std::vector<std::vector<TTStubRef>>>& stubs) const {
    for (auto ttModule = handle->begin(); ttModule != handle->end(); ttModule++) {
      // get det id for this module
      const DetId detId = ttModule->detId() + 1;
      // corresponding sensor module
      const SensorModule* sm = setup_->sensorModule(detId);
      // empty stub collection
      std::vector<TTStubRef>& ttStubRefs = stubs[sm->dtcId()][sm->modId()];
      ttStubRefs.reserve(ttModule->size());
      for (auto ttStub = ttModule->begin(); ttStub != ttModule->end(); ttStub++)
        ttStubRefs.emplace_back(makeRefTo(handle, ttStub));
    }
  }

  // board level transforming and routing of stubs
  void ProducerDTC::produce(const std::vector<std::vector<TTStubRef>>& dtc, TTDTC& ttDTC) const {
    const std::vector<const SensorModule*>& sms = setup_->dtcModules(dtcId_);
    const bool gig10 = (dtcId_ % setup_->sysNumATCASlot()) < setup_->sysSlotLimitPS();
    const int max = (gig10 ? setup_->cicNumStub10g() : setup_->cicNumStub5g()) * setup_->smNumCIC();
    // count number of stubs on this dtc
    auto acc = [](int sum, const std::vector<TTStubRef>& ttStubRefs) { return sum + ttStubRefs.size(); };
    const int nStubs = std::accumulate(dtc.begin(), dtc.end(), 0, acc);
    std::vector<StubFE> stubsFE;
    stubsFE.reserve(nStubs);
    // helper to sort stubs
    auto byBend = [](const StubFE& lhs, const StubFE& rhs) { return std::abs(lhs.bend()) < std::abs(rhs.bend()); };
    // convert TTStubs to front end stubs
    for (int modId = 0; modId < setup_->dtcNumModule(); modId++) {
      const std::vector<TTStubRef>& ttStubRefs = dtc[modId];
      if (ttStubRefs.empty())
        continue;
      // Module which produced this ttStubRefs
      const SensorModule* sm = sms[modId];
      // store stubs
      const auto begin = stubsFE.end();
      for (const TTStubRef& ttStubRef : ttStubRefs)
        stubsFE.emplace_back(setup_, sm, ttStubRef);
      // sort stubs by bend
      std::sort(begin, stubsFE.end(), byBend);
      // truncate if desired
      if (setup_->enableTruncation()) {
        const int size = std::distance(begin, stubsFE.end());
        if (size > max)
          stubsFE.erase(std::next(begin, max), stubsFE.end());
      }
    }
    // convert front end stubs to gloabl stubs
    std::vector<StubGL> stubsGL;
    stubsGL.reserve(nStubs);
    for (const StubFE& stubFE : stubsFE)
      stubsGL.emplace_back(stubFE);
    // convert global stubs to output stubs
    std::vector<std::vector<StubDTC>> stubsDTC(setup_->sysNumOverlap());
    for (std::vector<StubDTC>& stubs : stubsDTC)
      stubs.reserve(nStubs);
    for (int overlap = 0; overlap < setup_->sysNumOverlap(); overlap++) {
      std::vector<StubDTC>& stubs = stubsDTC[overlap];
      for (const StubGL& stubGL : stubsGL) {
        if (!stubGL.valid() || !stubGL.overlap().test(overlap))
          continue;
        stubs.emplace_back(stubGL, overlap);
        if (!stubs.back().valid())
          stubs.pop_back();
      }
    }
    // truncate if desired
    if (setup_->enableTruncation()) {
      for (std::vector<StubDTC>& stubs : stubsDTC)
        if (static_cast<int>(stubs.size()) > setup_->sysNumFrames())
          stubs.erase(std::next(stubs.begin(), setup_->sysNumFrames()), stubs.end());
    }
    // fill product
    const int region = dtcId_ / setup_->regNumDTC();
    const int board = dtcId_ % setup_->regNumDTC();
    for (int overlap = 0; overlap < setup_->sysNumOverlap(); overlap++) {
      const std::vector<StubDTC>& stubs = stubsDTC[overlap];
      tt::StreamStub stream;
      stream.reserve(stubs.size());
      for (const StubDTC& stub : stubs)
        stream.push_back(stub.frame());
      ttDTC.setStream(region, board, overlap, stream);
    }
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::ProducerDTC);
