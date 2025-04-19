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

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerTQ
   *  \brief  Bit accurate emulation of the track quality BDT
   *  \author Thomas Schuh
   *  \date   2024, Aug
   */
  class ProducerTQ : public edm::stream::EDProducer<> {
  public:
    explicit ProducerTQ(const edm::ParameterSet&);
    ~ProducerTQ() override {}
    void produce(edm::Event&, const edm::EventSetup&) override;

  private:
    typedef TrackQuality::Track Track;
    // ED input token of kf stubs
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubs_;
    // ED input token of kf tracks
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracks_;
    // ED output token for tracks
    edm::EDPutTokenT<tt::StreamsTrack> edPutTokenTracks_;
    // ED output token for additional track variables
    edm::EDPutTokenT<tt::Streams> edPutTokenTracksAdd_;
    // ED output token for stubs
    edm::EDPutTokenT<tt::StreamsStub> edPutTokenStubs_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // TrackQuality token
    edm::ESGetToken<TrackQuality, DataFormatsRcd> esGetTokenTrackQuality_;
  };

  ProducerTQ::ProducerTQ(const edm::ParameterSet& iConfig) {
    const std::string& label = iConfig.getParameter<std::string>("InputLabelTQ");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<tt::StreamsStub>(edm::InputTag(label, branchStubs));
    edGetTokenTracks_ = consumes<tt::StreamsTrack>(edm::InputTag(label, branchTracks));
    edPutTokenTracks_ = produces<tt::StreamsTrack>(branchTracks);
    edPutTokenTracksAdd_ = produces<tt::Streams>(branchTracks);
    edPutTokenStubs_ = produces<tt::StreamsStub>(branchStubs);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenTrackQuality_ = esConsumes();
  }

  void ProducerTQ::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to determine Track Quality
    const TrackQuality* trackQuality = &iSetup.getData(esGetTokenTrackQuality_);
    auto valid = [](int sum, const tt::FrameTrack& frame) { return sum + (frame.first.isNull() ? 0 : 1); };
    // empty TQ product
    tt::StreamsTrack outputTracks(setup->numRegions());
    tt::Streams outputTracksAdd(setup->numRegions());
    tt::StreamsStub outputStubs(setup->numRegions() * setup->numLayers());
    // read in KF Product and produce TQ product
    const tt::StreamsStub& streamsStubs = iEvent.get(edGetTokenStubs_);
    const tt::StreamsTrack& streamsTracks = iEvent.get(edGetTokenTracks_);
    for (int region = 0; region < setup->numRegions(); region++) {
      // calculate track quality
      const int offsetLayer = region * setup->numLayers();
      const tt::StreamTrack& streamTrack = streamsTracks[region];
      const int nTracks = std::accumulate(streamTrack.begin(), streamTrack.end(), 0, valid);
      std::vector<Track> tracks;
      tracks.reserve(nTracks);
      std::vector<Track*> stream;
      stream.reserve(streamTrack.size());
      for (int frame = 0; frame < static_cast<int>(streamTrack.size()); frame++) {
        const tt::FrameTrack& frameTrack = streamTrack[frame];
        if (frameTrack.first.isNull()) {
          stream.push_back(nullptr);
          continue;
        }
        tt::StreamStub streamStub;
        streamStub.reserve(setup->numLayers());
        for (int layer = 0; layer < setup->numLayers(); layer++)
          streamStub.push_back(streamsStubs[offsetLayer + layer][frame]);
        tracks.emplace_back(frameTrack, streamStub, trackQuality);
        stream.push_back(&tracks.back());
      }
      // fill TQ product
      outputTracks[region].reserve(stream.size());
      outputTracksAdd[region].reserve(stream.size());
      for (int layer = 0; layer < setup->numLayers(); layer++)
        outputStubs[offsetLayer + layer].reserve(stream.size());
      for (Track* track : stream) {
        if (!track) {
          outputTracks[region].emplace_back(tt::FrameTrack());
          outputTracksAdd[region].emplace_back(tt::Frame());
          for (int layer = 0; layer < setup->numLayers(); layer++)
            outputStubs[offsetLayer + layer].emplace_back(tt::FrameStub());
          continue;
        }
        outputTracks[region].emplace_back(track->frameTrack_);
        outputTracksAdd[region].emplace_back(track->frame_);
        for (int layer = 0; layer < setup->numLayers(); layer++)
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
