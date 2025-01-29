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
#include "L1Trigger/TrackerTFP/interface/KalmanFilterFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "L1Trigger/TrackerTFP/interface/KalmanFilter.h"

#include <string>
#include <vector>
#include <utility>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerKF
   *  \brief  L1TrackTrigger Kamlan Filter emulator
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class ProducerKF : public stream::EDProducer<> {
  public:
    explicit ProducerKF(const ParameterSet&);
    ~ProducerKF() override {}

  private:
    typedef State::Stub Stub;
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    void endStream() override {
      if (printDebug_)
        kalmanFilterFormats_.endJob();
    }
    // ED input token of sf stubs and tracks
    EDGetTokenT<StreamsStub> edGetTokenStubs_;
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    // ED output token for accepted stubs and tracks
    EDPutTokenT<StreamsStub> edPutTokenStubs_;
    EDPutTokenT<StreamsTrack> edPutTokenTracks_;
    // ED output token for number of accepted and lost States
    EDPutTokenT<int> edPutTokenNumStatesAccepted_;
    EDPutTokenT<int> edPutTokenNumStatesTruncated_;
    // ED output token for chi2s in r-phi and r-z plane
    EDPutTokenT<vector<pair<double, double>>> edPutTokenChi2s_;
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
    // helper class to encode layer
    const LayerEncoding* layerEncoding_ = nullptr;
    // helper class to tune internal kf variables
    KalmanFilterFormats kalmanFilterFormats_;
    // print end job internal unused MSB
    bool printDebug_;
    // number of channels
    int numChannel_;
    // number of processing regions
    int numRegions_;
    // number of kf layers
    int numLayers_;
  };

  ProducerKF::ProducerKF(const ParameterSet& iConfig) : kalmanFilterFormats_(iConfig) {
    printDebug_ = iConfig.getParameter<bool>("PrintKFDebug");
    const string& label = iConfig.getParameter<string>("InputLabelKF");
    const string& branchStubs = iConfig.getParameter<string>("BranchStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchTracks");
    const string& branchTruncated = iConfig.getParameter<string>("BranchTruncated");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(label, branchStubs));
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(label, branchTracks));
    edPutTokenStubs_ = produces<StreamsStub>(branchStubs);
    edPutTokenTracks_ = produces<StreamsTrack>(branchTracks);
    edPutTokenNumStatesAccepted_ = produces<int>(branchTracks);
    edPutTokenNumStatesTruncated_ = produces<int>(branchTruncated);
    edPutTokenChi2s_ = produces<vector<pair<double, double>>>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenLayerEncoding_ = esConsumes<LayerEncoding, LayerEncodingRcd, Transition::BeginRun>();
  }

  void ProducerKF::beginRun(const Run& iRun, const EventSetup& iSetup) {
    setup_ = &iSetup.getData(esGetTokenSetup_);
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    layerEncoding_ = &iSetup.getData(esGetTokenLayerEncoding_);
    kalmanFilterFormats_.beginRun(dataFormats_);
    numChannel_ = dataFormats_->numChannel(Process::kf);
    numRegions_ = setup_->numRegions();
    numLayers_ = setup_->numLayers();
  }

  void ProducerKF::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty KF products
    StreamsStub acceptedStubs(numRegions_ * numChannel_ * numLayers_);
    StreamsTrack acceptedTracks(numRegions_ * numChannel_);
    int numStatesAccepted(0);
    int numStatesTruncated(0);
    deque<pair<double, double>> chi2s;
    // read in SF Product and produce KF product
    const StreamsStub& allStubs = iEvent.get(edGetTokenStubs_);
    const StreamsTrack& allTracks = iEvent.get(edGetTokenTracks_);
    // helper
    auto validFrameT = [](int sum, const FrameTrack& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    auto validFrameS = [](int sum, const FrameStub& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    auto putT = [](const vector<TrackKF*>& objects, StreamTrack& stream) {
      auto toFrame = [](TrackKF* object) { return object ? object->frame() : FrameTrack(); };
      stream.reserve(objects.size());
      transform(objects.begin(), objects.end(), back_inserter(stream), toFrame);
    };
    auto putS = [](const vector<StubKF*>& objects, StreamStub& stream) {
      auto toFrame = [](StubKF* object) { return object ? object->frame() : FrameStub(); };
      stream.reserve(objects.size());
      transform(objects.begin(), objects.end(), back_inserter(stream), toFrame);
    };
    for (int region = 0; region < numRegions_; region++) {
      const int offset = region * numChannel_;
      // count input objects
      int nTracks(0);
      int nStubs(0);
      for (int channel = 0; channel < numChannel_; channel++) {
        const int index = offset + channel;
        const int offsetStubs = index * numLayers_;
        const StreamTrack& tracks = allTracks[index];
        nTracks += accumulate(tracks.begin(), tracks.end(), 0, validFrameT);
        for (int layer = 0; layer < numLayers_; layer++) {
          const StreamStub& stubs = allStubs[offsetStubs + layer];
          nStubs += accumulate(stubs.begin(), stubs.end(), 0, validFrameS);
        }
      }
      // storage of input data
      vector<TrackCTB> tracksCTB;
      tracksCTB.reserve(nTracks);
      vector<Stub> stubs;
      stubs.reserve(nStubs);
      // h/w liked organized pointer to input data
      vector<vector<TrackCTB*>> regionTracks(numChannel_);
      vector<vector<Stub*>> regionStubs(numChannel_ * numLayers_);
      // read input data
      for (int channel = 0; channel < numChannel_; channel++) {
        const int index = offset + channel;
        const int offsetAll = index * numLayers_;
        const int offsetRegion = channel * numLayers_;
        const StreamTrack& streamTrack = allTracks[index];
        vector<TrackCTB*>& tracks = regionTracks[channel];
        tracks.reserve(streamTrack.size());
        for (const FrameTrack& frame : streamTrack) {
          TrackCTB* track = nullptr;
          if (frame.first.isNonnull()) {
            tracksCTB.emplace_back(frame, dataFormats_);
            track = &tracksCTB.back();
          }
          tracks.push_back(track);
        }
        for (int layer = 0; layer < numLayers_; layer++) {
          for (const FrameStub& frame : allStubs[offsetAll + layer]) {
            Stub* stub = nullptr;
            if (frame.first.isNonnull()) {
              stubs.emplace_back(&kalmanFilterFormats_, frame);
              stub = &stubs.back();
            }
            regionStubs[offsetRegion + layer].push_back(stub);
          }
        }
      }
      // empty storage of output data
      vector<TrackKF> tracksKF;
      tracksKF.reserve(nTracks);
      vector<StubKF> stubsKF;
      stubsKF.reserve(nStubs);
      // object to fit tracks in a processing region
      KalmanFilter kf(setup_, dataFormats_, layerEncoding_, &kalmanFilterFormats_, tracksKF, stubsKF);
      // empty h/w liked organized pointer to output data
      vector<vector<TrackKF*>> streamsTrack(numChannel_);
      vector<vector<vector<StubKF*>>> streamsStub(numChannel_, vector<vector<StubKF*>>(numLayers_));
      // fill output products
      kf.produce(regionTracks, regionStubs, streamsTrack, streamsStub, numStatesAccepted, numStatesTruncated, chi2s);
      // convert data to ed products
      for (int channel = 0; channel < numChannel_; channel++) {
        const int index = offset + channel;
        const int offsetStubs = index * numLayers_;
        putT(streamsTrack[channel], acceptedTracks[index]);
        for (int layer = 0; layer < numLayers_; layer++)
          putS(streamsStub[channel][layer], acceptedStubs[offsetStubs + layer]);
      }
    }
    // store products
    iEvent.emplace(edPutTokenStubs_, std::move(acceptedStubs));
    iEvent.emplace(edPutTokenTracks_, std::move(acceptedTracks));
    iEvent.emplace(edPutTokenNumStatesAccepted_, numStatesAccepted);
    iEvent.emplace(edPutTokenNumStatesTruncated_, numStatesTruncated);
    iEvent.emplace(edPutTokenChi2s_, chi2s.begin(), chi2s.end());
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerKF);
