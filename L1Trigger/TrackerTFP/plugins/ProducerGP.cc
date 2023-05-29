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
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/GeometricProcessor.h"

#include <numeric>
#include <string>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerGP
   *  \brief  L1TrackTrigger Geometric Processor emulator
   *  \author Thomas Schuh
   *  \date   2020, March
   */
  class ProducerGP : public stream::EDProducer<> {
  public:
    explicit ProducerGP(const ParameterSet&);
    ~ProducerGP() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    virtual void endJob() {}

    // ED input token of DTC stubs
    EDGetTokenT<TTDTC> edGetToken_;
    // ED output token for accepted stubs
    EDPutTokenT<StreamsStub> edPutTokenAccepted_;
    // ED output token for lost stubs
    EDPutTokenT<StreamsStub> edPutTokenLost_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // configuration
    ParameterSet iConfig_;
    // helper classe to store configurations
    const Setup* setup_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
  };

  ProducerGP::ProducerGP(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const string& label = iConfig.getParameter<string>("LabelDTC");
    const string& branchAccepted = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchLost = iConfig.getParameter<string>("BranchLostStubs");
    // book in- and output ED products
    edGetToken_ = consumes<TTDTC>(InputTag(label, branchAccepted));
    edPutTokenAccepted_ = produces<StreamsStub>(branchAccepted);
    edPutTokenLost_ = produces<StreamsStub>(branchLost);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
  }

  void ProducerGP::beginRun(const Run& iRun, const EventSetup& iSetup) {
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
  }

  void ProducerGP::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty GP products
    StreamsStub accepted(dataFormats_->numStreams(Process::gp));
    StreamsStub lost(dataFormats_->numStreams(Process::gp));
    // read in DTC Product and produce TFP product
    if (setup_->configurationSupported()) {
      Handle<TTDTC> handle;
      iEvent.getByToken<TTDTC>(edGetToken_, handle);
      const TTDTC& ttDTC = *handle.product();
      for (int region = 0; region < setup_->numRegions(); region++) {
        // object to route Stubs of one region to one stream per sector
        GeometricProcessor gp(iConfig_, setup_, dataFormats_, region);
        // read in and organize input product
        gp.consume(ttDTC);
        // fill output products
        gp.produce(accepted, lost);
      }
    }
    // store products
    iEvent.emplace(edPutTokenAccepted_, std::move(accepted));
    iEvent.emplace(edPutTokenLost_, std::move(lost));
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerGP);
