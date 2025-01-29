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

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerCTB
   *  \brief  clean HT tracks and rrestructures them
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class ProducerCTB : public stream::EDProducer<> {
  public:
    explicit ProducerCTB(const ParameterSet&);
    ~ProducerCTB() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    // ED input token of Stubs
    EDGetTokenT<StreamsStub> edGetToken_;
    // ED output token for TTTracks
    EDPutTokenT<TTTracks> edPutTokenTTTracks_;
    // ED output token for stubs
    EDPutTokenT<StreamsStub> edPutTokenStubs_;
    // ED output token for tracks
    EDPutTokenT<StreamsTrack> edPutTokenTracks_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    ESGetToken<LayerEncoding, LayerEncodingRcd> esGetTokenLayerEncoding_;
    // helper class to store configurations
    const Setup* setup_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
    //
    const LayerEncoding* layerEncoding_ = nullptr;
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

  ProducerCTB::ProducerCTB(const ParameterSet& iConfig) {
    const string& label = iConfig.getParameter<string>("InputLabelCTB");
    const string& branchStubs = iConfig.getParameter<string>("BranchStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchTracks");
    // book in- and output ED products
    edGetToken_ = consumes<StreamsStub>(InputTag(label, branchStubs));
    edPutTokenStubs_ = produces<StreamsStub>(branchStubs);
    edPutTokenTTTracks_ = produces<TTTracks>(branchTracks);
    edPutTokenTracks_ = produces<StreamsTrack>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenLayerEncoding_ = esConsumes<LayerEncoding, LayerEncodingRcd, Transition::BeginRun>();
  }

  void ProducerCTB::beginRun(const Run& iRun, const EventSetup& iSetup) {
    setup_ = &iSetup.getData(esGetTokenSetup_);
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    layerEncoding_ = &iSetup.getData(esGetTokenLayerEncoding_);
    numChannelIn_ = dataFormats_->numChannel(Process::ht);
    numChannelOut_ = dataFormats_->numChannel(Process::ctb);
    numRegions_ = setup_->numRegions();
    numLayers_ = setup_->numLayers();
    // create data format for cot(theta)
    const double baseZ = dataFormats_->base(Variable::z, Process::ctb);
    const double baseR = dataFormats_->base(Variable::r, Process::ctb);
    const double range = dataFormats_->range(Variable::cot, Process::kf);
    const int baseShift = ceil(log2(range / baseZ * baseR / setup_->ctbNumBinsCot()));
    const int width = ceil(log2(setup_->ctbNumBinsCot()));
    const double base = baseZ / baseR * pow(2, baseShift);
    cot_ = DataFormat(true, width, base, range);
  }

  void ProducerCTB::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty output products
    StreamsTrack acceptedTracks(numRegions_ * numChannelOut_);
    StreamsStub acceptedStubs(numRegions_ * numChannelOut_ * numLayers_);
    vector<vector<deque<TrackCTB*>>> streamsTracks(numRegions_, vector<deque<TrackCTB*>>(numChannelOut_));
    vector<vector<vector<deque<StubCTB*>>>> streamsStubs(
        numRegions_, vector<vector<deque<StubCTB*>>>(numChannelOut_, vector<deque<StubCTB*>>(numLayers_)));
    // read input Product and produce output product
    const StreamsStub& streamsStub = iEvent.get(edGetToken_);
    // count stubs
    int nStubsHT(0);
    auto validFrame = [](int sum, const FrameStub& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    for (const StreamStub& stream : streamsStub)
      nStubsHT += accumulate(stream.begin(), stream.end(), 0, validFrame);
    // create input objects and count tracks
    vector<StubHT> stubsHT;
    stubsHT.reserve(nStubsHT);
    // count stubs
    int nTracksHT(0);
    for (const StreamStub& stream : streamsStub) {
      pair<int, int> trackId({setup_->htNumBinsPhiT(), setup_->gpNumBinsZT()});
      for (const FrameStub& frame : stream) {
        if (frame.first.isNull())
          continue;
        stubsHT.emplace_back(frame, dataFormats_);
        StubHT* stub = &stubsHT.back();
        if (trackId.first != stub->phiT() || trackId.second != stub->zT()) {
          nTracksHT++;
          trackId = {stub->phiT(), stub->zT()};
        }
      }
    }
    // object to clean and restructure tracks
    vector<StubCTB> stubsCTB;
    vector<TrackCTB> tracksCTB;
    tracksCTB.reserve(nTracksHT);
    stubsCTB.reserve(nStubsHT);
    CleanTrackBuilder ctb(setup_, dataFormats_, layerEncoding_, cot_, stubsCTB, tracksCTB);
    int iStub(0);
    for (int region = 0; region < numRegions_; region++) {
      const int offsetIn = region * numChannelIn_;
      const int offsetOut = region * numChannelOut_;
      // read h/w liked organized pointer to input data
      vector<vector<StubHT*>> streamsIn(numChannelIn_);
      for (int channelIn = 0; channelIn < numChannelIn_; channelIn++) {
        const StreamStub& channelStubs = streamsStub[offsetIn + channelIn];
        vector<StubHT*>& stream = streamsIn[channelIn];
        stream.reserve(channelStubs.size());
        for (const FrameStub& frame : channelStubs)
          stream.push_back(frame.first.isNull() ? nullptr : &stubsHT[iStub++]);
      }
      // empty h/w liked organized pointer to output data
      vector<deque<TrackCTB*>>& regionTracks = streamsTracks[region];
      vector<vector<deque<StubCTB*>>>& regionStubs = streamsStubs[region];
      // fill output data
      ctb.produce(streamsIn, regionTracks, regionStubs);
      // fill ed stubs
      for (int channelOut = 0; channelOut < numChannelOut_; channelOut++) {
        const int offset = (offsetOut + channelOut) * numLayers_;
        const vector<deque<StubCTB*>>& channelStubs = regionStubs[channelOut];
        for (int layer = 0; layer < numLayers_; layer++) {
          StreamStub& accepted = acceptedStubs[offset + layer];
          const deque<StubCTB*>& layerStubs = channelStubs[layer];
          accepted.reserve(layerStubs.size());
          for (StubCTB* stub : layerStubs)
            accepted.emplace_back(stub ? stub->frame() : FrameStub());
        }
      }
    }
    // store TTTracks
    int nTracks(0);
    auto valid = [](int sum, TrackCTB* track) { return sum += (track ? 1 : 0); };
    for (const vector<deque<TrackCTB*>>& region : streamsTracks)
      for (const deque<TrackCTB*>& channel : region)
        nTracks += accumulate(channel.begin(), channel.end(), 0, valid);
    TTTracks ttTracks;
    ttTracks.reserve(nTracks);
    for (int region = 0; region < numRegions_; region++) {
      const vector<deque<TrackCTB*>>& regionTracks = streamsTracks[region];
      const vector<vector<deque<StubCTB*>>>& regionStubs = streamsStubs[region];
      for (int channelOut = 0; channelOut < numChannelOut_; channelOut++) {
        const deque<TrackCTB*>& channelTracks = regionTracks[channelOut];
        const vector<deque<StubCTB*>>& channelStubs = regionStubs[channelOut];
        for (int frame = 0; frame < (int)channelTracks.size(); frame++) {
          TrackCTB* track = channelTracks[frame];
          if (!track)
            continue;
          const auto begin = next(channelTracks.begin(), frame);
          const auto end = find_if(begin + 1, channelTracks.end(), [](TrackCTB* track) { return track; });
          const int size = distance(begin, end);
          vector<vector<StubCTB*>> stubs(numLayers_);
          for (int layer = 0; layer < numLayers_; layer++) {
            const deque<StubCTB*>& layerStubs = channelStubs[layer];
            vector<StubCTB*>& layerTrack = stubs[layer];
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
    const OrphanHandle<TTTracks> handle = iEvent.emplace(edPutTokenTTTracks_, std::move(ttTracks));
    // add TTTrackRefs
    int iTrk(0);
    int iChan(0);
    for (const vector<deque<TrackCTB*>>& region : streamsTracks) {
      for (const deque<TrackCTB*>& stream : region) {
        StreamTrack& streamTrack = acceptedTracks[iChan++];
        for (TrackCTB* track : stream) {
          if (!track) {
            streamTrack.emplace_back(FrameTrack());
            continue;
          }
          FrameTrack frame = track->frame();
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