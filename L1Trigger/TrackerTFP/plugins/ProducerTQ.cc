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
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/TrackQuality.h"

#include <string>
#include <numeric>

using namespace std;
using namespace edm;
using namespace tt;
using namespace trackerTFP;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerTQ
   *  \brief  Bit accurate emulation of the track quality BDT
   *  \author Thomas Schuh
   *  \date   2024, Aug
   */
  class ProducerTQ : public stream::EDProducer<> {
  public:
    explicit ProducerTQ(const ParameterSet&);
    ~ProducerTQ() override {}
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;

  private:
    typedef TrackQuality::Track Track;
    // ED input token of kf stubs
    EDGetTokenT<StreamsStub> edGetTokenStubs_;
    // ED input token of kf tracks
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    // ED output token for tracks
    EDPutTokenT<StreamsTrack> edPutTokenTracks_;
    // ED output token for additional track variables
    EDPutTokenT<Streams> edPutTokenTracksAdd_;
    // ED output token for stubs
    EDPutTokenT<StreamsStub> edPutTokenStubs_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // TrackQuality token
    ESGetToken<TrackQuality, DataFormatsRcd> esGetTokenTrackQuality_;
    // number of processing regions
    int numRegions_;
    // number of kf layers
    int numLayers_;
  };

  ProducerTQ::ProducerTQ(const ParameterSet& iConfig) {
    const string& label = iConfig.getParameter<string>("InputLabelTQ");
    const string& branchStubs = iConfig.getParameter<string>("BranchStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchTracks");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(label, branchStubs));
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(label, branchTracks));
    edPutTokenTracks_ = produces<StreamsTrack>(branchTracks);
    edPutTokenTracksAdd_ = produces<Streams>(branchTracks);
    edPutTokenStubs_ = produces<StreamsStub>(branchStubs);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenTrackQuality_ = esConsumes();
  }

  void ProducerTQ::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    const Setup* setup = &iSetup.getData(esGetTokenSetup_);
    numRegions_ = setup->numRegions();
    numLayers_ = setup->numLayers();
  }

  void ProducerTQ::produce(Event& iEvent, const EventSetup& iSetup) {
    // helper class to determine Track Quality
    const TrackQuality* trackQuality = &iSetup.getData(esGetTokenTrackQuality_);
    auto valid = [](int sum, const FrameTrack& frame) { return sum += (frame.first.isNull() ? 0 : 1); };
    // empty TQ product
    StreamsTrack outputTracks(numRegions_);
    Streams outputTracksAdd(numRegions_);
    StreamsStub outputStubs(numRegions_ * numLayers_);
    // read in KF Product and produce TQ product
    Handle<StreamsStub> handleStubs;
    iEvent.getByToken<StreamsStub>(edGetTokenStubs_, handleStubs);
    const StreamsStub& streamsStubs = *handleStubs.product();
    Handle<StreamsTrack> handleTracks;
    iEvent.getByToken<StreamsTrack>(edGetTokenTracks_, handleTracks);
    const StreamsTrack& streamsTracks = *handleTracks.product();
    for (int region = 0; region < numRegions_; region++) {
      // calculate track quality
      const int offsetLayer = region * numLayers_;
      const StreamTrack& streamTrack = streamsTracks[region];
      const int nTracks = accumulate(streamTrack.begin(), streamTrack.end(), 0, valid);
      vector<Track> tracks;
      tracks.reserve(nTracks);
      vector<Track*> stream;
      stream.reserve(streamTrack.size());
      for (int frame = 0; frame < (int)streamTrack.size(); frame++) {
        const FrameTrack& frameTrack = streamTrack[frame];
        if (frameTrack.first.isNull()) {
          stream.push_back(nullptr);
          continue;
        }
        StreamStub streamStub;
        streamStub.reserve(numLayers_);
        for (int layer = 0; layer < numLayers_; layer++)
          streamStub.push_back(streamsStubs[offsetLayer + layer][frame]);
        tracks.emplace_back(frameTrack, streamStub, trackQuality);
        stream.push_back(&tracks.back());
      }
      // fill TQ product
      outputTracks[region].reserve(stream.size());
      outputTracksAdd[region].reserve(stream.size());
      for (int layer = 0; layer < numLayers_; layer++)
        outputStubs[offsetLayer + layer].reserve(stream.size());
      for (Track* track : stream) {
        if (!track) {
          outputTracks[region].emplace_back(FrameTrack());
          outputTracksAdd[region].emplace_back(Frame());
          for (int layer = 0; layer < numLayers_; layer++)
            outputStubs[offsetLayer + layer].emplace_back(FrameStub());
          continue;
        }
        outputTracks[region].emplace_back(track->frameTrack_);
        outputTracksAdd[region].emplace_back(track->frame_);
        for (int layer = 0; layer < numLayers_; layer++)
          outputStubs[offsetLayer + layer].emplace_back(track->streamStub_[layer]);
      }
    }
    // store TQ product
    iEvent.emplace(edPutTokenTracks_, std::move(outputTracks));
    iEvent.emplace(edPutTokenTracksAdd_, std::move(outputTracksAdd));
    iEvent.emplace(edPutTokenStubs_, streamsStubs);
  }
}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerTQ);
