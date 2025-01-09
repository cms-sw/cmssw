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
    // configuration
    ParameterSet iConfig_;
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
  };

  ProducerKF::ProducerKF(const ParameterSet& iConfig) : iConfig_(iConfig), kalmanFilterFormats_(iConfig) {
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
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to encode layer
    layerEncoding_ = &iSetup.getData(esGetTokenLayerEncoding_);
    // helper class to tune internal kf variables
    kalmanFilterFormats_.beginRun(dataFormats_);
  }

  void ProducerKF::produce(Event& iEvent, const EventSetup& iSetup) {
    static const int numChannel = dataFormats_->numChannel(Process::kf);
    static const int numRegions = setup_->numRegions();
    static const int numLayers = setup_->numLayers();
    // empty KF products
    StreamsStub acceptedStubs(numRegions * numChannel * numLayers);
    StreamsTrack acceptedTracks(numRegions * numChannel);
    int numStatesAccepted(0);
    int numStatesTruncated(0);
    deque<pair<double, double>> chi2s;
    // read in SF Product and produce KF product
    Handle<StreamsStub> handleStubs;
    iEvent.getByToken<StreamsStub>(edGetTokenStubs_, handleStubs);
    const StreamsStub& allStubs = *handleStubs;
    Handle<StreamsTrack> handleTracks;
    iEvent.getByToken<StreamsTrack>(edGetTokenTracks_, handleTracks);
    const StreamsTrack& allTracks = *handleTracks;
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
    for (int region = 0; region < numRegions; region++) {
      const int offset = region * numChannel;
      // count input objects
      int nTracks(0);
      int nStubs(0);
      for (int channel = 0; channel < numChannel; channel++) {
        const int index = offset + channel;
        const int offsetStubs = index * numLayers;
        const StreamTrack& tracks = allTracks[index];
        nTracks += accumulate(tracks.begin(), tracks.end(), 0, validFrameT);
        for (int layer = 0; layer < numLayers; layer++) {
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
      vector<vector<TrackCTB*>> regionTracks(numChannel);
      vector<vector<Stub*>> regionStubs(numChannel * numLayers);
      // read input data
      for (int channel = 0; channel < numChannel; channel++) {
        const int index = offset + channel;
        const int offsetAll = index * numLayers;
        const int offsetRegion = channel * numLayers;
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
        for (int layer = 0; layer < numLayers; layer++) {
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
      KalmanFilter kf(iConfig_, setup_, dataFormats_, layerEncoding_, &kalmanFilterFormats_, tracksKF, stubsKF);
      // empty h/w liked organized pointer to output data
      vector<vector<TrackKF*>> streamsTrack(numChannel);
      vector<vector<vector<StubKF*>>> streamsStub(numChannel, vector<vector<StubKF*>>(numLayers));
      // fill output products
      kf.produce(regionTracks, regionStubs, streamsTrack, streamsStub, numStatesAccepted, numStatesTruncated, chi2s);
      // convert data to ed products
      for (int channel = 0; channel < numChannel; channel++) {
        const int index = offset + channel;
        const int offsetStubs = index * numLayers;
        putT(streamsTrack[channel], acceptedTracks[index]);
        for (int layer = 0; layer < numLayers; layer++)
          putS(streamsStub[channel][layer], acceptedStubs[offsetStubs + layer]);
      }
    }
    // store products
    iEvent.emplace(edPutTokenStubs_, move(acceptedStubs));
    iEvent.emplace(edPutTokenTracks_, move(acceptedTracks));
    iEvent.emplace(edPutTokenNumStatesAccepted_, numStatesAccepted);
    iEvent.emplace(edPutTokenNumStatesTruncated_, numStatesTruncated);
    iEvent.emplace(edPutTokenChi2s_, chi2s.begin(), chi2s.end());
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerKF);
