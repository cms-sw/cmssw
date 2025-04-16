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
    ~ProducerCTB() override {}

  private:
    void beginRun(const edm::Run&, const edm::EventSetup&) override;
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
    //
    DataFormat cot_;
    // number of inpit channel
    int numChannelIn_;
    // number of output channel
    int numChannelOut_;
    // number of processing regions
    int numRegions_;
    // number of kf layers
    int numLayers_;
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

  void ProducerCTB::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    numChannelIn_ = dataFormats->numChannel(Process::ht);
    numChannelOut_ = dataFormats->numChannel(Process::ctb);
    numRegions_ = setup->numRegions();
    numLayers_ = setup->numLayers();
    // create data format for cot(theta)
    const double baseZ = dataFormats->base(Variable::z, Process::ctb);
    const double baseR = dataFormats->base(Variable::r, Process::ctb);
    const double range = dataFormats->range(Variable::cot, Process::kf);
    const int baseShift = std::ceil(std::log2(range / baseZ * baseR / setup->ctbNumBinsCot()));
    const int width = std::ceil(std::log2(setup->ctbNumBinsCot()));
    const double base = baseZ / baseR * pow(2, baseShift);
    cot_ = DataFormat(true, width, base, range);
  }

  void ProducerCTB::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    const LayerEncoding* layerEncoding = &iSetup.getData(esGetTokenLayerEncoding_);
    // empty output products
    tt::StreamsTrack acceptedTracks(numRegions_ * numChannelOut_);
    tt::StreamsStub acceptedStubs(numRegions_ * numChannelOut_ * numLayers_);
    std::vector<std::vector<std::deque<TrackCTB*>>> streamsTracks(numRegions_,
                                                                  std::vector<std::deque<TrackCTB*>>(numChannelOut_));
    std::vector<std::vector<std::vector<std::deque<StubCTB*>>>> streamsStubs(
        numRegions_,
        std::vector<std::vector<std::deque<StubCTB*>>>(numChannelOut_, std::vector<std::deque<StubCTB*>>(numLayers_)));
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
    CleanTrackBuilder ctb(setup, dataFormats, layerEncoding, cot_, stubsCTB, tracksCTB);
    int iStub(0);
    for (int region = 0; region < numRegions_; region++) {
      const int offsetIn = region * numChannelIn_;
      const int offsetOut = region * numChannelOut_;
      // read h/w liked organized pointer to input data
      std::vector<std::vector<StubHT*>> streamsIn(numChannelIn_);
      for (int channelIn = 0; channelIn < numChannelIn_; channelIn++) {
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
      for (int channelOut = 0; channelOut < numChannelOut_; channelOut++) {
        const int offset = (offsetOut + channelOut) * numLayers_;
        const std::vector<std::deque<StubCTB*>>& channelStubs = regionStubs[channelOut];
        for (int layer = 0; layer < numLayers_; layer++) {
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
    for (int region = 0; region < numRegions_; region++) {
      const std::vector<std::deque<TrackCTB*>>& regionTracks = streamsTracks[region];
      const std::vector<std::vector<std::deque<StubCTB*>>>& regionStubs = streamsStubs[region];
      for (int channelOut = 0; channelOut < numChannelOut_; channelOut++) {
        const std::deque<TrackCTB*>& channelTracks = regionTracks[channelOut];
        const std::vector<std::deque<StubCTB*>>& channelStubs = regionStubs[channelOut];
        for (int frame = 0; frame < static_cast<int>(channelTracks.size()); frame++) {
          TrackCTB* track = channelTracks[frame];
          if (!track)
            continue;
          const auto begin = std::next(channelTracks.begin(), frame);
          const auto end = std::find_if(begin + 1, channelTracks.end(), [](TrackCTB* track) { return track; });
          const int size = std::distance(begin, end);
          std::vector<std::vector<StubCTB*>> stubs(numLayers_);
          for (int layer = 0; layer < numLayers_; layer++) {
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
