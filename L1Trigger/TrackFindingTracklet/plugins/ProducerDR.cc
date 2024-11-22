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
#include "L1Trigger/TrackFindingTracklet/interface/DR.h"
#include "SimDataFormats/Associations/interface/TTTypes.h"

#include <string>
#include <vector>
#include <deque>
#include <iterator>
#include <cmath>
#include <numeric>

using namespace std;
using namespace edm;
using namespace tt;
using namespace trackerTFP;

namespace trklet {

  /*! \class  trklet::ProducerDR
   *  \brief  Emulates removal of duplicated TTTracks f/w.
   *          Track order determined by TrackMultiplexer affects performance
   *  \author Thomas Schuh
   *  \date   2023, Feb
   */
  class ProducerDR : public stream::EDProducer<> {
  public:
    explicit ProducerDR(const ParameterSet&);
    ~ProducerDR() override {}

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
    // LayerEncoding token
    ESGetToken<LayerEncoding, LayerEncodingRcd> esGetTokenLayerEncoding_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // ChannelAssignment token
    ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_ = nullptr;
    // helper class to encode layer
    const LayerEncoding* layerEncoding_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
    // helper class to assign tracks to channel
    const ChannelAssignment* channelAssignment_ = nullptr;
  };

  ProducerDR::ProducerDR(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const string& label = iConfig.getParameter<string>("InputLabelDR");
    const string& branchStubs = iConfig.getParameter<string>("BranchStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchTracks");
    // book in- and output ED products
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(label, branchTracks));
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(label, branchStubs));
    edPutTokenTracks_ = produces<StreamsTrack>(branchTracks);
    edPutTokenStubs_ = produces<StreamsStub>(branchStubs);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenLayerEncoding_ = esConsumes<LayerEncoding, LayerEncodingRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenChannelAssignment_ = esConsumes<ChannelAssignment, ChannelAssignmentRcd, Transition::BeginRun>();
  }

  void ProducerDR::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    // helper class to encode layer
    layerEncoding_ = &iSetup.getData(esGetTokenLayerEncoding_);
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to assign tracks to channel
    channelAssignment_ = &iSetup.getData(esGetTokenChannelAssignment_);
  }

  void ProducerDR::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty DR products
    const int numStreamsTracks = setup_->numRegions();
    const int numStreamsStubs = numStreamsTracks * setup_->numLayers();
    StreamsStub streamsStub(numStreamsStubs);
    StreamsTrack streamsTrack(numStreamsTracks);
    // read in TBout Product and produce KFin product
    Handle<StreamsStub> handleStubs;
    iEvent.getByToken<StreamsStub>(edGetTokenStubs_, handleStubs);
    const StreamsStub& stubs = *handleStubs;
    Handle<StreamsTrack> handleTracks;
    iEvent.getByToken<StreamsTrack>(edGetTokenTracks_, handleTracks);
    const StreamsTrack& tracks = *handleTracks;
    for (int region = 0; region < setup_->numRegions(); region++) {
      // object to remove duplicated tracks in a processing region
      DuplicateRemoval dr(iConfig_, setup_, layerEncoding_, dataFormats_, channelAssignment_, region);
      // read in and organize input tracks and stubs
      dr.consume(tracks, stubs);
      // fill output products
      dr.produce(streamsTrack, streamsStub);
    }
    // store products
    iEvent.emplace(edPutTokenStubs_, move(streamsStub));
    iEvent.emplace(edPutTokenTracks_, move(streamsTrack));
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerDR);
