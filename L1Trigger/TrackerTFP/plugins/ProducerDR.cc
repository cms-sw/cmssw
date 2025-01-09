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

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/DuplicateRemoval.h"

#include <string>
#include <numeric>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerDR
   *  \brief  L1TrackTrigger duplicate removal emulator
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
    void endStream() override {}
    // ED input token of sf stubs and tracks
    EDGetTokenT<StreamsStub> edGetTokenStubs_;
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    // ED output token for accepted stubs and tracks
    EDPutTokenT<StreamsStub> edPutTokenStubs_;
    EDPutTokenT<StreamsTrack> edPutTokenTracks_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
  };

  ProducerDR::ProducerDR(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const string& label = iConfig.getParameter<string>("InputLabelDR");
    const string& branchStubs = iConfig.getParameter<string>("BranchStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchTracks");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(label, branchStubs));
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(label, branchTracks));
    edPutTokenStubs_ = produces<StreamsStub>(branchStubs);
    edPutTokenTracks_ = produces<StreamsTrack>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
  }

  void ProducerDR::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
  }

  void ProducerDR::produce(Event& iEvent, const EventSetup& iSetup) {
    static const int numChannelIn = dataFormats_->numChannel(Process::kf);
    static const int numChannelOut = dataFormats_->numChannel(Process::dr);
    static const int numRegions = setup_->numRegions();
    static const int numLayers = setup_->numLayers();
    // empty DR products
    StreamsStub acceptedStubs(numRegions * numChannelOut * numLayers);
    StreamsTrack acceptedTracks(numRegions * numChannelOut);
    // read in KF Product and produce DR product
    Handle<StreamsStub> handleStubs;
    iEvent.getByToken<StreamsStub>(edGetTokenStubs_, handleStubs);
    const StreamsStub& allStubs = *handleStubs;
    Handle<StreamsTrack> handleTracks;
    iEvent.getByToken<StreamsTrack>(edGetTokenTracks_, handleTracks);
    const StreamsTrack& allTracks = *handleTracks;
    // helper
    auto validFrameT = [](int sum, const FrameTrack& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    auto validFrameS = [](int sum, const FrameStub& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    auto putT = [](const vector<TrackDR*>& objects, StreamTrack& stream) {
      auto toFrame = [](TrackDR* object) { return object ? object->frame() : FrameTrack(); };
      stream.reserve(objects.size());
      transform(objects.begin(), objects.end(), back_inserter(stream), toFrame);
    };
    auto putS = [](const vector<StubDR*>& objects, StreamStub& stream) {
      auto toFrame = [](StubDR* object) { return object ? object->frame() : FrameStub(); };
      stream.reserve(objects.size());
      transform(objects.begin(), objects.end(), back_inserter(stream), toFrame);
    };
    for (int region = 0; region < numRegions; region++) {
      const int offsetIn = region * numChannelIn;
      const int offsetOut = region * numChannelOut;
      // count input objects
      int nTracks(0);
      int nStubs(0);
      for (int channelIn = 0; channelIn < numChannelIn; channelIn++) {
        const int index = offsetIn + channelIn;
        const int offset = index * numLayers;
        const StreamTrack& tracks = allTracks[index];
        nTracks += accumulate(tracks.begin(), tracks.end(), 0, validFrameT);
        for (int layer = 0; layer < numLayers; layer++) {
          const StreamStub& stubs = allStubs[offset + layer];
          nStubs += accumulate(stubs.begin(), stubs.end(), 0, validFrameS);
        }
      }
      // storage of input data
      vector<TrackKF> tracksKF;
      tracksKF.reserve(nTracks);
      vector<StubKF> stubsKF;
      stubsKF.reserve(nStubs);
      // h/w liked organized pointer to input data
      vector<vector<TrackKF*>> regionTracks(numChannelIn);
      vector<vector<StubKF*>> regionStubs(numChannelIn * numLayers);
      // read input data
      for (int channelIn = 0; channelIn < numChannelIn; channelIn++) {
        const int index = offsetIn + channelIn;
        const int offsetAll = index * numLayers;
        const int offsetRegion = channelIn * numLayers;
        const StreamTrack& streamTrack = allTracks[index];
        vector<TrackKF*>& tracks = regionTracks[channelIn];
        tracks.reserve(streamTrack.size());
        for (const FrameTrack& frame : streamTrack) {
          TrackKF* track = nullptr;
          if (frame.first.isNonnull()) {
            tracksKF.emplace_back(frame, dataFormats_);
            track = &tracksKF.back();
          }
          tracks.push_back(track);
        }
        for (int layer = 0; layer < numLayers; layer++) {
          for (const FrameStub& frame : allStubs[offsetAll + layer]) {
            StubKF* stub = nullptr;
            if (frame.first.isNonnull()) {
              stubsKF.emplace_back(frame, dataFormats_);
              stub = &stubsKF.back();
            }
            regionStubs[offsetRegion + layer].push_back(stub);
          }
        }
      }
      // empty storage of output data
      vector<TrackDR> tracksDR;
      tracksDR.reserve(nTracks);
      vector<StubDR> stubsDR;
      stubsDR.reserve(nStubs);
      // object to remove duplicates in a processing region
      DuplicateRemoval dr(iConfig_, setup_, dataFormats_, tracksDR, stubsDR);
      // empty h/w liked organized pointer to output data
      vector<vector<TrackDR*>> streamsTrack(numChannelOut);
      vector<vector<StubDR*>> streamsStub(numChannelOut * numLayers);
      // fill output data
      dr.produce(regionTracks, regionStubs, streamsTrack, streamsStub);
      // convert data to ed products
      for (int channelOut = 0; channelOut < numChannelOut; channelOut++) {
        const int index = offsetOut + channelOut;
        const int offsetRegion = channelOut * numLayers;
        const int offsetAll = index * numLayers;
        putT(streamsTrack[channelOut], acceptedTracks[index]);
        for (int layer = 0; layer < numLayers; layer++)
          putS(streamsStub[offsetRegion + layer], acceptedStubs[offsetAll + layer]);
      }
    }
    // store products
    iEvent.emplace(edPutTokenStubs_, move(acceptedStubs));
    iEvent.emplace(edPutTokenTracks_, move(acceptedTracks));
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerDR);