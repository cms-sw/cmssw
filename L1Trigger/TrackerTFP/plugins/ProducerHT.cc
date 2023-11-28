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

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/HoughTransform.h"

#include <string>
#include <numeric>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerHT
   *  \brief  L1TrackTrigger Hough Transform emulator
   *  \author Thomas Schuh
   *  \date   2020, March
   */
  class ProducerHT : public stream::EDProducer<> {
  public:
    explicit ProducerHT(const ParameterSet&);
    ~ProducerHT() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    virtual void endJob() {}

    // ED input token of gp stubs
    EDGetTokenT<StreamsStub> edGetToken_;
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
    // helper class to store configurations
    const Setup* setup_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
  };

  ProducerHT::ProducerHT(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const string& label = iConfig.getParameter<string>("LabelGP");
    const string& branchAccepted = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchLost = iConfig.getParameter<string>("BranchLostStubs");
    // book in- and output ED products
    edGetToken_ = consumes<StreamsStub>(InputTag(label, branchAccepted));
    edPutTokenAccepted_ = produces<StreamsStub>(branchAccepted);
    edPutTokenLost_ = produces<StreamsStub>(branchLost);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
  }

  void ProducerHT::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
  }

  void ProducerHT::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty HT products
    StreamsStub accepted(dataFormats_->numStreams(Process::ht));
    StreamsStub lost(dataFormats_->numStreams(Process::ht));
    // read in DTC Product and produce TFP product
    if (setup_->configurationSupported()) {
      Handle<StreamsStub> handle;
      iEvent.getByToken<StreamsStub>(edGetToken_, handle);
      const StreamsStub& streams = *handle.product();
      for (int region = 0; region < setup_->numRegions(); region++) {
        // object to find initial rough candidates in r-phi in a region
        HoughTransform ht(iConfig_, setup_, dataFormats_, region);
        // read in and organize input product
        ht.consume(streams);
        // fill output products
        ht.produce(accepted, lost);
      }
    }
    // store products
    iEvent.emplace(edPutTokenAccepted_, std::move(accepted));
    iEvent.emplace(edPutTokenLost_, std::move(lost));
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerHT);
