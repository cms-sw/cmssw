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
    void produce(Event&, const EventSetup&) override;

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
    ESGetToken<DataFormats, ChannelAssignmentRcd> esGetTokenDataFormats_;
    // ChannelAssignment token
    ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
    // helper class to store tracklet configurations
    Settings settings_;
  };

  ProducerTM::ProducerTM(const ParameterSet& iConfig) {
    const string& label = iConfig.getParameter<string>("InputLabelTM");
    const string& branchStubs = iConfig.getParameter<string>("BranchStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchTracks");
    // book in- and output ED products
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(label, branchTracks));
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(label, branchStubs));
    edPutTokenStubs_ = produces<StreamsStub>(branchStubs);
    edPutTokenTracks_ = produces<StreamsTrack>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
    esGetTokenChannelAssignment_ = esConsumes();
  }

  void ProducerTM::produce(Event& iEvent, const EventSetup& iSetup) {
    // helper class to store configurations
    const Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to assign tracks to channel
    const ChannelAssignment* channelAssignment = &iSetup.getData(esGetTokenChannelAssignment_);
    // empty TM products
    StreamsStub streamsStub(setup->numRegions() * channelAssignment->tmNumLayers());
    StreamsTrack streamsTrack(setup->numRegions());
    // read in TBout Product and produce TM product
    const StreamsStub& stubs = iEvent.get(edGetTokenStubs_);
    const StreamsTrack& tracks = iEvent.get(edGetTokenTracks_);
    for (int region = 0; region < setup->numRegions(); region++) {
      // object to reformat tracks from tracklet fromat to TMTT format in a processing region
      TrackMultiplexer tm(setup, dataFormats, channelAssignment, &settings_, region);
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
