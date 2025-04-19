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
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"
#include "L1Trigger/TrackFindingTracklet/interface/DuplicateRemoval.h"
#include "SimDataFormats/Associations/interface/TTTypes.h"

#include <string>
#include <vector>
#include <deque>
#include <iterator>
#include <cmath>
#include <numeric>

namespace trklet {

  /*! \class  trklet::ProducerDR
   *  \brief  Emulates removal of duplicated TTTracks f/w.
   *          Track order determined by TrackMultiplexer affects performance
   *  \author Thomas Schuh
   *  \date   2023, Feb
   */
  class ProducerDR : public edm::stream::EDProducer<> {
  public:
    explicit ProducerDR(const edm::ParameterSet&);
    ~ProducerDR() override {}

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
    // LayerEncoding token
    edm::ESGetToken<trackerTFP::LayerEncoding, trackerTFP::DataFormatsRcd> esGetTokenLayerEncoding_;
    // DataFormats token
    edm::ESGetToken<DataFormats, ChannelAssignmentRcd> esGetTokenDataFormats_;
    // ChannelAssignment token
    edm::ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
  };

  ProducerDR::ProducerDR(const edm::ParameterSet& iConfig) {
    const std::string& label = iConfig.getParameter<std::string>("InputLabelDR");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    // book in- and output ED products
    edGetTokenTracks_ = consumes<tt::StreamsTrack>(edm::InputTag(label, branchTracks));
    edGetTokenStubs_ = consumes<tt::StreamsStub>(edm::InputTag(label, branchStubs));
    edPutTokenTracks_ = produces<tt::StreamsTrack>(branchTracks);
    edPutTokenStubs_ = produces<tt::StreamsStub>(branchStubs);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenLayerEncoding_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
    esGetTokenChannelAssignment_ = esConsumes();
  }

  void ProducerDR::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to encode layer
    const trackerTFP::LayerEncoding* layerEncoding = &iSetup.getData(esGetTokenLayerEncoding_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to assign tracks to channel
    const ChannelAssignment* channelAssignment = &iSetup.getData(esGetTokenChannelAssignment_);
    // empty DR products
    tt::StreamsStub streamsStub(setup->numRegions() * setup->numLayers());
    tt::StreamsTrack streamsTrack(setup->numRegions());
    // read in TBout Product and produce KFin product
    const tt::StreamsStub& stubs = iEvent.get(edGetTokenStubs_);
    const tt::StreamsTrack& tracks = iEvent.get(edGetTokenTracks_);
    for (int region = 0; region < setup->numRegions(); region++) {
      // object to remove duplicated tracks in a processing region
      DuplicateRemoval dr(setup, layerEncoding, dataFormats, channelAssignment, region);
      // read in and organize input tracks and stubs
      dr.consume(tracks, stubs);
      // fill output products
      dr.produce(streamsTrack, streamsStub);
    }
    // store products
    iEvent.emplace(edPutTokenStubs_, std::move(streamsStub));
    iEvent.emplace(edPutTokenTracks_, std::move(streamsTrack));
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerDR);
