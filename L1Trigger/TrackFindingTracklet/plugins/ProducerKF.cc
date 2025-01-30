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
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "L1Trigger/TrackFindingTracklet/interface/KalmanFilter.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"

#include <string>
#include <vector>

using namespace std;
using namespace edm;
using namespace tt;
using namespace tmtt;

namespace trklet {

  /*! \class  trklet::ProducerKF
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
        kalmanFilterFormats_.endJob();
    }
    // ED input token of sf stubs and tracks
    EDGetTokenT<StreamsStub> edGetTokenStubs_;
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    // ED output token for accepted stubs and tracks
    EDPutTokenT<TTTracks> edPutTokenTTTracks_;
    EDPutTokenT<StreamsStub> edPutTokenStubs_;
    EDPutTokenT<StreamsTrack> edPutTokenTracks_;
    // ED output token for number of accepted and lost States
    EDPutTokenT<int> edPutTokenNumStatesAccepted_;
    EDPutTokenT<int> edPutTokenNumStatesTruncated_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, ChannelAssignmentRcd> esGetTokenDataFormats_;
    // provides dataformats of Kalman filter internals
    KalmanFilterFormats kalmanFilterFormats_;
    //
    Settings settings_;
    //
    KFParamsComb tmtt4_;
    //
    KFParamsComb tmtt5_;
    //
    KFParamsComb* tmtt_;
    // print end job internal unused MSB
    bool printDebug_;
    //
    bool use5ParameterFit_;
  };

  ProducerKF::ProducerKF(const ParameterSet& iConfig)
      : kalmanFilterFormats_(iConfig),
        settings_(iConfig),
        tmtt4_(&settings_, 4, "KF5ParamsComb"),
        tmtt5_(&settings_, 5, "KF4ParamsComb"),
        tmtt_(&tmtt4_) {
    printDebug_ = iConfig.getParameter<bool>("PrintKFDebug");
    use5ParameterFit_ = iConfig.getParameter<bool>("Use5ParameterFit");
    if (use5ParameterFit_)
      tmtt_ = &tmtt5_;
    const string& label = iConfig.getParameter<string>("InputLabelKF");
    const string& branchStubs = iConfig.getParameter<string>("BranchStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchTracks");
    const string& branchTruncated = iConfig.getParameter<string>("BranchTruncated");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(label, branchStubs));
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(label, branchTracks));
    edPutTokenStubs_ = produces<StreamsStub>(branchStubs);
    edPutTokenTracks_ = produces<StreamsTrack>(branchTracks);
    edPutTokenTTTracks_ = produces<TTTracks>(branchTracks);
    edPutTokenNumStatesAccepted_ = produces<int>(branchTracks);
    edPutTokenNumStatesTruncated_ = produces<int>(branchTruncated);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, ChannelAssignmentRcd, Transition::BeginRun>();
  }

  void ProducerKF::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    const Setup* setup = &iSetup.getData(esGetTokenSetup_);
    settings_.setMagneticField(setup->bField());
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // provides dataformats of Kalman filter internals
    kalmanFilterFormats_.consume(dataFormats);
  }

  void ProducerKF::produce(Event& iEvent, const EventSetup& iSetup) {
    // helper class to store configurations
    const Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    auto valid = [](int sum, const FrameTrack& f) { return sum += (f.first.isNull() ? 0 : 1); };
    // empty KF products
    StreamsStub streamsStub(setup->numRegions() * setup->numLayers());
    StreamsTrack streamsTrack(setup->numRegions());
    int numStatesAccepted(0);
    int numStatesTruncated(0);
    // read in DR Product and produce KF product
    const StreamsStub& stubs = iEvent.get(edGetTokenStubs_);
    const StreamsTrack& tracks = iEvent.get(edGetTokenTracks_);
    // prep TTTracks
    TTTracks ttTracks;
    vector<TTTrackRef> ttTrackRefs;
    if (use5ParameterFit_) {
      int nTracks(0);
      for (const StreamTrack& stream : tracks)
        nTracks += accumulate(stream.begin(), stream.end(), 0, valid);
      ttTracks.reserve(nTracks);
      ttTrackRefs.reserve(nTracks);
      for (const StreamTrack& stream : tracks)
        for (const FrameTrack& frame : stream)
          if (frame.first.isNonnull())
            ttTrackRefs.push_back(frame.first);
    }
    for (int region = 0; region < setup->numRegions(); region++) {
      // object to fit tracks in a processing region
      KalmanFilter kf(setup, dataFormats, &kalmanFilterFormats_, &settings_, tmtt_, region, ttTracks);
      // read in and organize input tracks and stubs
      kf.consume(tracks, stubs);
      // fill output products
      kf.produce(streamsStub, streamsTrack, numStatesAccepted, numStatesTruncated);
    }
    if (use5ParameterFit_) {
      // store ttTracks
      const OrphanHandle<TTTracks> oh = iEvent.emplace(edPutTokenTTTracks_, std::move(ttTracks));
      // replace ttTrackRefs in track streams
      int iTrk(0);
      for (StreamTrack& stream : streamsTrack)
        for (FrameTrack& frame : stream)
          if (frame.first.isNonnull())
            frame.first = TTTrackRef(oh, iTrk++);
    }
    // store products
    iEvent.emplace(edPutTokenStubs_, std::move(streamsStub));
    iEvent.emplace(edPutTokenTracks_, std::move(streamsTrack));
    iEvent.emplace(edPutTokenNumStatesAccepted_, numStatesAccepted);
    iEvent.emplace(edPutTokenNumStatesTruncated_, numStatesTruncated);
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerKF);
