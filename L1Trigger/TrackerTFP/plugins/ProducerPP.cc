#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <string>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerPP
   *  \brief  L1TrackTrigger PatchPanel between DTC and TFP emulator
   *  \author Thomas Schuh
   *  \date   2023, April
   */
  class ProducerPP : public stream::EDProducer<> {
  public:
    explicit ProducerPP(const ParameterSet&);
    ~ProducerPP() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    virtual void endJob() {}
    // ED input token of DTC stubs
    EDGetTokenT<TTDTC> edGetToken_;
    // ED output token for accepted stubs
    EDPutTokenT<StreamsStub> edPutToken_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // configuration
    ParameterSet iConfig_;
    // helper classe to store configurations
    const Setup* setup_ = nullptr;
  };

  ProducerPP::ProducerPP(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const string& label = iConfig.getParameter<string>("InputLabelPP");
    const string& branch = iConfig.getParameter<string>("BranchStubs");
    // book in- and output ED products
    edGetToken_ = consumes<TTDTC>(InputTag(label, branch));
    edPutToken_ = produces<StreamsStub>(branch);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
  }

  void ProducerPP::beginRun(const Run& iRun, const EventSetup& iSetup) { setup_ = &iSetup.getData(esGetTokenSetup_); }

  void ProducerPP::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty GP products
    StreamsStub stubs(setup_->numRegions() * setup_->numDTCsPerTFP());
    // read in DTC Product and produce TFP product
    Handle<TTDTC> handle;
    iEvent.getByToken<TTDTC>(edGetToken_, handle);
    const TTDTC& ttDTC = *handle.product();
    for (int region = 0; region < setup_->numRegions(); region++) {
      const int offset = region * setup_->numDTCsPerTFP();
      for (int channel = 0; channel < setup_->numDTCsPerTFP(); channel++)
        stubs[offset + channel] = ttDTC.stream(region, channel);
    }
    // store products
    iEvent.emplace(edPutToken_, move(stubs));
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerPP);