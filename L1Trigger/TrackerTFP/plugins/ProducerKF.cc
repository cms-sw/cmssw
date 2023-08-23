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
#include "L1Trigger/TrackerTFP/interface/KalmanFilterFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "L1Trigger/TrackerTFP/interface/KalmanFilter.h"

#include <string>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerKF
   *  \brief  L1TrackTrigger Kamlan Filter emulator
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class ProducerKF : public stream::EDProducer<> {
  public:
    explicit ProducerKF(const ParameterSet&);
    ~ProducerKF() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    void endStream() override {
      if (printDebug_)
        kalmanFilterFormats_->endJob();
    }

    // ED input token of sf stubs and tracks
    EDGetTokenT<StreamsStub> edGetTokenStubs_;
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    // ED output token for accepted stubs and tracks
    EDPutTokenT<StreamsStub> edPutTokenAcceptedStubs_;
    EDPutTokenT<StreamsTrack> edPutTokenAcceptedTracks_;
    // ED output token for lost stubs and tracks
    EDPutTokenT<StreamsStub> edPutTokenLostStubs_;
    EDPutTokenT<StreamsTrack> edPutTokenLostTracks_;
    // ED output token for number of accepted and lost States
    EDPutTokenT<int> edPutTokenNumAcceptedStates_;
    EDPutTokenT<int> edPutTokenNumLostStates_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // KalmanFilterFormats token
    ESGetToken<KalmanFilterFormats, KalmanFilterFormatsRcd> esGetTokenKalmanFilterFormats_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
    // helper class to
    KalmanFilterFormats* kalmanFilterFormats_ = nullptr;
    // print end job internal unused MSB
    bool printDebug_;
  };

  ProducerKF::ProducerKF(const ParameterSet& iConfig) : iConfig_(iConfig) {
    printDebug_ = iConfig.getParameter<bool>("PrintKFDebug");
    const string& label = iConfig.getParameter<string>("LabelKFin");
    const string& branchAcceptedStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchAcceptedTracks = iConfig.getParameter<string>("BranchAcceptedTracks");
    const string& branchLostStubs = iConfig.getParameter<string>("BranchLostStubs");
    const string& branchLostTracks = iConfig.getParameter<string>("BranchLostTracks");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(label, branchAcceptedStubs));
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(label, branchAcceptedTracks));
    edPutTokenAcceptedStubs_ = produces<StreamsStub>(branchAcceptedStubs);
    edPutTokenAcceptedTracks_ = produces<StreamsTrack>(branchAcceptedTracks);
    edPutTokenLostStubs_ = produces<StreamsStub>(branchLostStubs);
    edPutTokenLostTracks_ = produces<StreamsTrack>(branchLostTracks);
    edPutTokenNumAcceptedStates_ = produces<int>(branchAcceptedTracks);
    edPutTokenNumLostStates_ = produces<int>(branchLostTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenKalmanFilterFormats_ = esConsumes<KalmanFilterFormats, KalmanFilterFormatsRcd, Transition::BeginRun>();
  }

  void ProducerKF::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to
    kalmanFilterFormats_ = const_cast<KalmanFilterFormats*>(&iSetup.getData(esGetTokenKalmanFilterFormats_));
  }

  void ProducerKF::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty KF products
    StreamsStub acceptedStubs(dataFormats_->numStreamsStubs(Process::kf));
    StreamsTrack acceptedTracks(dataFormats_->numStreamsTracks(Process::kf));
    StreamsStub lostStubs(dataFormats_->numStreamsStubs(Process::kf));
    StreamsTrack lostTracks(dataFormats_->numStreamsTracks(Process::kf));
    int numAcceptedStates(0);
    int numLostStates(0);
    // read in SF Product and produce KF product
    if (setup_->configurationSupported()) {
      Handle<StreamsStub> handleStubs;
      iEvent.getByToken<StreamsStub>(edGetTokenStubs_, handleStubs);
      const StreamsStub& stubs = *handleStubs;
      Handle<StreamsTrack> handleTracks;
      iEvent.getByToken<StreamsTrack>(edGetTokenTracks_, handleTracks);
      const StreamsTrack& tracks = *handleTracks;
      for (int region = 0; region < setup_->numRegions(); region++) {
        // object to fit tracks in a processing region
        KalmanFilter kf(iConfig_, setup_, dataFormats_, kalmanFilterFormats_, region);
        // read in and organize input tracks and stubs
        kf.consume(tracks, stubs);
        // fill output products
        kf.produce(acceptedStubs, acceptedTracks, lostStubs, lostTracks, numAcceptedStates, numLostStates);
      }
    }
    // store products
    iEvent.emplace(edPutTokenAcceptedStubs_, std::move(acceptedStubs));
    iEvent.emplace(edPutTokenAcceptedTracks_, std::move(acceptedTracks));
    iEvent.emplace(edPutTokenLostStubs_, std::move(lostStubs));
    iEvent.emplace(edPutTokenLostTracks_, std::move(lostTracks));
    iEvent.emplace(edPutTokenNumAcceptedStates_, numAcceptedStates);
    iEvent.emplace(edPutTokenNumLostStates_, numLostStates);
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerKF);
