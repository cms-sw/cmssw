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

namespace trklet {

  /*! \class  trklet::ProducerTM
   *  \brief  Transforms format of Track Builder into that expected by DR input and muxes all channels to 1.
              Since DR keeps first tracks the mux ordering (currently from low seed id to high seed id) is important.
   *  \author Thomas Schuh
   *  \date   2023, Jan
   */
  class ProducerTM : public edm::stream::EDProducer<> {
  public:
    explicit ProducerTM(const edm::ParameterSet&);
    ~ProducerTM() override {}

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;

    // ED input token of Tracks
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracks_;
    // ED input token of Stubs
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubs_;
    // ED output token for stubs
    edm::EDPutTokenT<tt::StreamsStub> edPutTokenStubs_;
    // ED output token for tracks
    edm::EDPutTokenT<tt::StreamsTrack> edPutTokenTracks_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, ChannelAssignmentRcd> esGetTokenDataFormats_;
    // ChannelAssignment token
    edm::ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
    // helper class to store tracklet configurations
    trklet::Settings settings_;
  };

  ProducerTM::ProducerTM(const edm::ParameterSet& iConfig) {
    const std::string& label = iConfig.getParameter<std::string>("InputLabelTM");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    // book in- and output ED products
    edGetTokenTracks_ = consumes<tt::StreamsTrack>(edm::InputTag(label, branchTracks));
    edGetTokenStubs_ = consumes<tt::StreamsStub>(edm::InputTag(label, branchStubs));
    edPutTokenStubs_ = produces<tt::StreamsStub>(branchStubs);
    edPutTokenTracks_ = produces<tt::StreamsTrack>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
    esGetTokenChannelAssignment_ = esConsumes();
  }

  void ProducerTM::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to assign tracks to channel
    const ChannelAssignment* channelAssignment = &iSetup.getData(esGetTokenChannelAssignment_);
    // empty TM products
    tt::StreamsStub streamsStub(setup->numRegions() * channelAssignment->tmNumLayers());
    tt::StreamsTrack streamsTrack(setup->numRegions());
    // read in TBout Product and produce TM product
    const tt::StreamsStub& stubs = iEvent.get(edGetTokenStubs_);
    const tt::StreamsTrack& tracks = iEvent.get(edGetTokenTracks_);
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
