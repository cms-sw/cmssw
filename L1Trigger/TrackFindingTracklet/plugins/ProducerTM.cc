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

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackMultiplexer.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <string>
#include <vector>
#include <deque>
#include <iterator>
#include <cmath>
#include <numeric>

using namespace std;
using namespace edm;
using namespace tt;

namespace trklet {

  /*! \class  trklet::ProducerTM
   *  \brief  Transforms format of Track Builder into that expected by DR input and muxes all channels to 1.
              Since DR keeps first tracks the mux ordering (currently from low seed id to high seed id) is important.
   *  \author Thomas Schuh
   *  \date   2023, Jan
   */
  class ProducerTM : public stream::EDProducer<> {
  public:
    explicit ProducerTM(const ParameterSet&);
    ~ProducerTM() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    virtual void endJob() {}

    // ED input token of Tracks
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    // ED input token of Stubs
    EDGetTokenT<StreamsStub> edGetTokenStubs_;
    // ED output token for stubs
    EDPutTokenT<StreamsStub> edPutTokenStubs_;
    // ED output token for tracks
    EDPutTokenT<StreamsTrack> edPutTokenTracks_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // ChannelAssignment token
    ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
    // helper class to assign tracks to channel
    const ChannelAssignment* channelAssignment_ = nullptr;
    // helper class to store tracklet configurations
    Settings settings_;
  };

  ProducerTM::ProducerTM(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const string& label = iConfig.getParameter<string>("InputLabelTM");
    const string& branchStubs = iConfig.getParameter<string>("BranchStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchTracks");
    // book in- and output ED products
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(label, branchTracks));
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(label, branchStubs));
    edPutTokenStubs_ = produces<StreamsStub>(branchStubs);
    edPutTokenTracks_ = produces<StreamsTrack>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenChannelAssignment_ = esConsumes<ChannelAssignment, ChannelAssignmentRcd, Transition::BeginRun>();
  }

  void ProducerTM::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to assign tracks to channel
    channelAssignment_ = &iSetup.getData(esGetTokenChannelAssignment_);
  }

  void ProducerTM::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty TM products
    const int numStreamsTracks = setup_->numRegions();
    const int numStreamsStubs = numStreamsTracks * channelAssignment_->tmNumLayers();
    StreamsStub streamsStub(numStreamsStubs);
    StreamsTrack streamsTrack(numStreamsTracks);
    // read in TBout Product and produce TM product
    Handle<StreamsStub> handleStubs;
    iEvent.getByToken<StreamsStub>(edGetTokenStubs_, handleStubs);
    const StreamsStub& stubs = *handleStubs;
    Handle<StreamsTrack> handleTracks;
    iEvent.getByToken<StreamsTrack>(edGetTokenTracks_, handleTracks);
    const StreamsTrack& tracks = *handleTracks;
    for (int region = 0; region < setup_->numRegions(); region++) {
      // object to reformat tracks from tracklet fromat to TMTT format in a processing region
      TrackMultiplexer tm(iConfig_, setup_, dataFormats_, channelAssignment_, &settings_, region);
      // read in and organize input tracks and stubs
      tm.consume(tracks, stubs);
      // fill output products
      tm.produce(streamsTrack, streamsStub);
    }
    // store products
    iEvent.emplace(edPutTokenTracks_, std::move(streamsTrack));
    iEvent.emplace(edPutTokenStubs_, std::move(streamsStub));
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerTM);
