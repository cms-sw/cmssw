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
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/CleanTrackBuilder.h"

#include <string>
#include <vector>
#include <deque>
#include <iterator>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerCTB
   *  \brief  clean HT tracks and rrestructures them
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class ProducerCTB : public edm::stream::EDProducer<> {
  public:
    explicit ProducerCTB(const edm::ParameterSet&);
    ~ProducerCTB() override = default;

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;
    // ED input token of Stubs
    edm::EDGetTokenT<tt::StreamsStub> edGetToken_;
    // ED output token for TTTracks
    edm::EDPutTokenT<tt::TTTracks> edPutTokenTTTracks_;
    // ED output token for stubs
    edm::EDPutTokenT<tt::StreamsStub> edPutTokenStubs_;
    // ED output token for tracks
    edm::EDPutTokenT<tt::StreamsTrack> edPutTokenTracks_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    edm::ESGetToken<LayerEncoding, DataFormatsRcd> esGetTokenLayerEncoding_;
  };

  ProducerCTB::ProducerCTB(const edm::ParameterSet& iConfig) {
    const std::string& label = iConfig.getParameter<std::string>("InputLabelCTB");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    // book in- and output ED products
    edGetToken_ = consumes<tt::StreamsStub>(edm::InputTag(label, branchStubs));
    edPutTokenStubs_ = produces<tt::StreamsStub>(branchStubs);
    edPutTokenTTTracks_ = produces<tt::TTTracks>(branchTracks);
    edPutTokenTracks_ = produces<tt::StreamsTrack>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
    esGetTokenLayerEncoding_ = esConsumes();
  }

  void ProducerCTB::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    const LayerEncoding* layerEncoding = &iSetup.getData(esGetTokenLayerEncoding_);
    // empty output products
    tt::StreamsTrack acceptedTracks(setup->numRegions() * dataFormats->numChannel(Process::ctb));
    tt::StreamsStub acceptedStubs(setup->numRegions() * dataFormats->numChannel(Process::ctb) * setup->numLayers());
    std::vector<std::vector<std::deque<TrackCTB*>>> streamsTracks(
        setup->numRegions(), std::vector<std::deque<TrackCTB*>>(dataFormats->numChannel(Process::ctb)));
    std::vector<std::vector<std::vector<std::deque<StubCTB*>>>> streamsStubs(
        setup->numRegions(),
        std::vector<std::vector<std::deque<StubCTB*>>>(dataFormats->numChannel(Process::ctb),
                                                       std::vector<std::deque<StubCTB*>>(setup->numLayers())));
    // read input Product and produce output product
    const tt::StreamsStub& streamsStub = iEvent.get(edGetToken_);
    // count stubs
    int nStubsHT(0);
    auto validFrame = [](int sum, const tt::FrameStub& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    for (const tt::StreamStub& stream : streamsStub)
      nStubsHT += std::accumulate(stream.begin(), stream.end(), 0, validFrame);
    // create input objects and count tracks
    std::vector<StubHT> stubsHT;
    stubsHT.reserve(nStubsHT);
    // count stubs
    int nTracksHT(0);
    for (const tt::StreamStub& stream : streamsStub) {
      std::pair<int, int> trackId({setup->htNumBinsPhiT(), setup->gpNumBinsZT()});
      for (const tt::FrameStub& frame : stream) {
        if (frame.first.isNull())
          continue;
        stubsHT.emplace_back(frame, dataFormats);
        StubHT* stub = &stubsHT.back();
        if (trackId.first != stub->phiT() || trackId.second != stub->zT()) {
          nTracksHT++;
          trackId = {stub->phiT(), stub->zT()};
        }
      }
    }
    // object to clean and restructure tracks
    std::vector<StubCTB> stubsCTB;
    stubsCTB.reserve(nStubsHT);
    std::vector<TrackCTB> tracksCTB;
    tracksCTB.reserve(nTracksHT);
    CleanTrackBuilder ctb(setup, dataFormats, layerEncoding, stubsCTB, tracksCTB);
    int iStub(0);
    for (int region = 0; region < setup->numRegions(); region++) {
      const int offsetIn = region * dataFormats->numChannel(Process::ht);
      const int offsetOut = region * dataFormats->numChannel(Process::ctb);
      // read h/w liked organized pointer to input data
      std::vector<std::vector<StubHT*>> streamsIn(dataFormats->numChannel(Process::ht));
      for (int channelIn = 0; channelIn < dataFormats->numChannel(Process::ht); channelIn++) {
        const tt::StreamStub& channelStubs = streamsStub[offsetIn + channelIn];
        std::vector<StubHT*>& stream = streamsIn[channelIn];
        stream.reserve(channelStubs.size());
        for (const tt::FrameStub& frame : channelStubs)
          stream.push_back(frame.first.isNull() ? nullptr : &stubsHT[iStub++]);
      }
      // empty h/w liked organized pointer to output data
      std::vector<std::deque<TrackCTB*>>& regionTracks = streamsTracks[region];
      std::vector<std::vector<std::deque<StubCTB*>>>& regionStubs = streamsStubs[region];
      // fill output data
      ctb.produce(streamsIn, regionTracks, regionStubs);
      // fill ed stubs
      for (int channelOut = 0; channelOut < dataFormats->numChannel(Process::ctb); channelOut++) {
        const int offset = (offsetOut + channelOut) * setup->numLayers();
        const std::vector<std::deque<StubCTB*>>& channelStubs = regionStubs[channelOut];
        for (int layer = 0; layer < setup->numLayers(); layer++) {
          tt::StreamStub& accepted = acceptedStubs[offset + layer];
          const std::deque<StubCTB*>& layerStubs = channelStubs[layer];
          accepted.reserve(layerStubs.size());
          for (StubCTB* stub : layerStubs)
            accepted.emplace_back(stub ? stub->frame() : tt::FrameStub());
        }
      }
    }
    // store TTTracks
    int nTracks(0);
    auto valid = [](int sum, TrackCTB* track) { return sum + (track ? 1 : 0); };
    for (const std::vector<std::deque<TrackCTB*>>& region : streamsTracks)
      for (const std::deque<TrackCTB*>& channel : region)
        nTracks += std::accumulate(channel.begin(), channel.end(), 0, valid);
    tt::TTTracks ttTracks;
    ttTracks.reserve(nTracks);
    for (int region = 0; region < setup->numRegions(); region++) {
      const std::vector<std::deque<TrackCTB*>>& regionTracks = streamsTracks[region];
      const std::vector<std::vector<std::deque<StubCTB*>>>& regionStubs = streamsStubs[region];
      for (int channelOut = 0; channelOut < dataFormats->numChannel(Process::ctb); channelOut++) {
        const std::deque<TrackCTB*>& channelTracks = regionTracks[channelOut];
        const std::vector<std::deque<StubCTB*>>& channelStubs = regionStubs[channelOut];
        for (int frame = 0; frame < static_cast<int>(channelTracks.size()); frame++) {
          TrackCTB* track = channelTracks[frame];
          if (!track)
            continue;
          const auto begin = std::next(channelTracks.begin(), frame);
          const auto end = std::find_if(begin + 1, channelTracks.end(), [](TrackCTB* track) { return track; });
          const int size = std::distance(begin, end);
          std::vector<std::vector<StubCTB*>> stubs(setup->numLayers());
          for (int layer = 0; layer < setup->numLayers(); layer++) {
            const std::deque<StubCTB*>& layerStubs = channelStubs[layer];
            std::vector<StubCTB*>& layerTrack = stubs[layer];
            layerTrack.reserve(size);
            for (int s = 0; s < size; s++) {
              StubCTB* stub = layerStubs[frame + s];
              if (stub)
                layerTrack.push_back(stub);
            }
          }
          ctb.put(track, stubs, region, ttTracks);
        }
      }
    }
    const edm::OrphanHandle<tt::TTTracks> handle = iEvent.emplace(edPutTokenTTTracks_, std::move(ttTracks));
    // add TTTrackRefs
    int iTrk(0);
    int iChan(0);
    for (const std::vector<std::deque<TrackCTB*>>& region : streamsTracks) {
      for (const std::deque<TrackCTB*>& stream : region) {
        tt::StreamTrack& streamTrack = acceptedTracks[iChan++];
        for (TrackCTB* track : stream) {
          if (!track) {
            streamTrack.emplace_back(tt::FrameTrack());
            continue;
          }
          tt::FrameTrack frame = track->frame();
          frame.first = TTTrackRef(handle, iTrk++);
          streamTrack.emplace_back(frame);
        }
      }
    }
    // store tracks
    iEvent.emplace(edPutTokenTracks_, std::move(acceptedTracks));
    // store stubs
    iEvent.emplace(edPutTokenStubs_, std::move(acceptedStubs));
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerCTB);
