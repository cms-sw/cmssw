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
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackTrigger/interface/SensorModule.h"
#include "L1Trigger/TrackerDTC/interface/LayerEncoding.h"
#include "L1Trigger/TrackerDTC/interface/DTC.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"

#include <numeric>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>

namespace trackerDTC {

  /*! \class  trackerDTC::ProducerDTC
   *  \brief  Class to produce hardware like structured TTStub Collection used by Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Jan
   */
  class ProducerDTC : public edm::stream::EDProducer<> {
  public:
    explicit ProducerDTC(const edm::ParameterSet&);
    ~ProducerDTC() override {}

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;
    // ED input token of TTStubs
    edm::EDGetTokenT<TTStubDetSetVec> edGetToken_;
    // ED output token for accepted stubs
    edm::EDPutTokenT<TTDTC> edPutTokenAccepted_;
    // ED output token for lost stubs
    edm::EDPutTokenT<TTDTC> edPutTokenLost_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<trackerTFP::DataFormats, trackerTFP::DataFormatsRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    edm::ESGetToken<LayerEncoding, tt::SetupRcd> esGetTokenLayerEncoding_;
  };

  ProducerDTC::ProducerDTC(const edm::ParameterSet& iConfig) {
    // book in- and output ED products
    const auto& inputTag = iConfig.getParameter<edm::InputTag>("InputTag");
    const auto& branchAccepted = iConfig.getParameter<std::string>("BranchAccepted");
    const auto& branchLost = iConfig.getParameter<std::string>("BranchLost");
    edGetToken_ = consumes<TTStubDetSetVec>(inputTag);
    edPutTokenAccepted_ = produces<TTDTC>(branchAccepted);
    edPutTokenLost_ = produces<TTDTC>(branchLost);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
    esGetTokenLayerEncoding_ = esConsumes();
  }

  void ProducerDTC::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const trackerTFP::DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // class to encode layer ids used between DTC and TFP in Hybrid
    const LayerEncoding* layerEncoding = &iSetup.getData(esGetTokenLayerEncoding_);
    // empty DTC products
    TTDTC productAccepted = setup->ttDTC();
    TTDTC productLost = setup->ttDTC();
    // read in stub collection
    edm::Handle<TTStubDetSetVec> handle;
    iEvent.getByToken(edGetToken_, handle);
    // apply cabling map, reorganise stub collections
    std::vector<std::vector<std::vector<TTStubRef>>> stubsDTCs(
        setup->numDTCs(), std::vector<std::vector<TTStubRef>>(setup->numModulesPerDTC()));
    for (auto module = handle->begin(); module != handle->end(); module++) {
      // DetSetVec->detId + 1 = tk layout det id
      const DetId detId = module->detId() + setup->offsetDetIdDSV();
      // corresponding sensor module
      tt::SensorModule* sm = setup->sensorModule(detId);
      // empty stub collection
      std::vector<TTStubRef>& stubsModule = stubsDTCs[sm->dtcId()][sm->modId()];
      stubsModule.reserve(module->size());
      for (TTStubDetSet::const_iterator ttStub = module->begin(); ttStub != module->end(); ttStub++)
        stubsModule.emplace_back(makeRefTo(handle, ttStub));
    }
    // board level processing
    for (int dtcId = 0; dtcId < setup->numDTCs(); dtcId++) {
      // create single outer tracker DTC board
      DTC dtc(setup, dataFormats, layerEncoding, dtcId, stubsDTCs.at(dtcId));
      // route stubs and fill products
      dtc.produce(productAccepted, productLost);
    }
    // store ED products
    iEvent.emplace(edPutTokenAccepted_, std::move(productAccepted));
    iEvent.emplace(edPutTokenLost_, std::move(productLost));
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::ProducerDTC);
