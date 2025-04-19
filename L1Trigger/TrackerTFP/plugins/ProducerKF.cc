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

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerKF
   *  \brief  L1TrackTrigger Kamlan Filter emulator
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class ProducerKF : public edm::stream::EDProducer<> {
  public:
    explicit ProducerKF(const edm::ParameterSet&);
    ~ProducerKF() override {}

  private:
    typedef State::Stub Stub;
    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void produce(edm::Event&, const edm::EventSetup&) override;
    void endStream() override {
      std::stringstream ss;
      if (printDebug_)
        kalmanFilterFormats_.endJob(ss);
      edm::LogPrint(moduleDescription().moduleName()) << ss.str();
    }
    // ED input token of sf stubs and tracks
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubs_;
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracks_;
    // ED output token for accepted stubs and tracks
    edm::EDPutTokenT<tt::StreamsStub> edPutTokenStubs_;
    edm::EDPutTokenT<tt::StreamsTrack> edPutTokenTracks_;
    // ED output token for number of accepted and lost States
    edm::EDPutTokenT<int> edPutTokenNumStatesAccepted_;
    edm::EDPutTokenT<int> edPutTokenNumStatesTruncated_;
    // ED output token for chi2s in r-phi and r-z plane
    edm::EDPutTokenT<std::vector<std::pair<double, double>>> edPutTokenChi2s_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    edm::ESGetToken<LayerEncoding, DataFormatsRcd> esGetTokenLayerEncoding_;
    // helper class to tune internal kf variables
    KalmanFilterFormats kalmanFilterFormats_;
    // KalmanFilterFormats configuraation
    ConfigKF iConfig_;
    // print end job internal unused MSB
    bool printDebug_;
    // number of channels
    int numChannel_;
    // number of processing regions
    int numRegions_;
    // number of kf layers
    int numLayers_;
  };

  ProducerKF::ProducerKF(const edm::ParameterSet& iConfig) {
    // KalmanFilterFormats configuraation
    iConfig_.enableIntegerEmulation_ = iConfig.getParameter<bool>("EnableIntegerEmulation");
    iConfig_.widthR00_ = iConfig.getParameter<int>("WidthR00");
    iConfig_.widthR11_ = iConfig.getParameter<int>("WidthR11");
    iConfig_.widthC00_ = iConfig.getParameter<int>("WidthC00");
    iConfig_.widthC01_ = iConfig.getParameter<int>("WidthC01");
    iConfig_.widthC11_ = iConfig.getParameter<int>("WidthC11");
    iConfig_.widthC22_ = iConfig.getParameter<int>("WidthC22");
    iConfig_.widthC23_ = iConfig.getParameter<int>("WidthC23");
    iConfig_.widthC33_ = iConfig.getParameter<int>("WidthC33");
    iConfig_.baseShiftx0_ = iConfig.getParameter<int>("BaseShiftx0");
    iConfig_.baseShiftx1_ = iConfig.getParameter<int>("BaseShiftx1");
    iConfig_.baseShiftx2_ = iConfig.getParameter<int>("BaseShiftx2");
    iConfig_.baseShiftx3_ = iConfig.getParameter<int>("BaseShiftx3");
    iConfig_.baseShiftr0_ = iConfig.getParameter<int>("BaseShiftr0");
    iConfig_.baseShiftr1_ = iConfig.getParameter<int>("BaseShiftr1");
    iConfig_.baseShiftS00_ = iConfig.getParameter<int>("BaseShiftS00");
    iConfig_.baseShiftS01_ = iConfig.getParameter<int>("BaseShiftS01");
    iConfig_.baseShiftS12_ = iConfig.getParameter<int>("BaseShiftS12");
    iConfig_.baseShiftS13_ = iConfig.getParameter<int>("BaseShiftS13");
    iConfig_.baseShiftR00_ = iConfig.getParameter<int>("BaseShiftR00");
    iConfig_.baseShiftR11_ = iConfig.getParameter<int>("BaseShiftR11");
    iConfig_.baseShiftInvR00Approx_ = iConfig.getParameter<int>("BaseShiftInvR00Approx");
    iConfig_.baseShiftInvR11Approx_ = iConfig.getParameter<int>("BaseShiftInvR11Approx");
    iConfig_.baseShiftInvR00Cor_ = iConfig.getParameter<int>("BaseShiftInvR00Cor");
    iConfig_.baseShiftInvR11Cor_ = iConfig.getParameter<int>("BaseShiftInvR11Cor");
    iConfig_.baseShiftInvR00_ = iConfig.getParameter<int>("BaseShiftInvR00");
    iConfig_.baseShiftInvR11_ = iConfig.getParameter<int>("BaseShiftInvR11");
    iConfig_.baseShiftS00Shifted_ = iConfig.getParameter<int>("BaseShiftS00Shifted");
    iConfig_.baseShiftS01Shifted_ = iConfig.getParameter<int>("BaseShiftS01Shifted");
    iConfig_.baseShiftS12Shifted_ = iConfig.getParameter<int>("BaseShiftS12Shifted");
    iConfig_.baseShiftS13Shifted_ = iConfig.getParameter<int>("BaseShiftS13Shifted");
    iConfig_.baseShiftK00_ = iConfig.getParameter<int>("BaseShiftK00");
    iConfig_.baseShiftK10_ = iConfig.getParameter<int>("BaseShiftK10");
    iConfig_.baseShiftK21_ = iConfig.getParameter<int>("BaseShiftK21");
    iConfig_.baseShiftK31_ = iConfig.getParameter<int>("BaseShiftK31");
    iConfig_.baseShiftC00_ = iConfig.getParameter<int>("BaseShiftC00");
    iConfig_.baseShiftC01_ = iConfig.getParameter<int>("BaseShiftC01");
    iConfig_.baseShiftC11_ = iConfig.getParameter<int>("BaseShiftC11");
    iConfig_.baseShiftC22_ = iConfig.getParameter<int>("BaseShiftC22");
    iConfig_.baseShiftC23_ = iConfig.getParameter<int>("BaseShiftC23");
    iConfig_.baseShiftC33_ = iConfig.getParameter<int>("BaseShiftC33");
    iConfig_.baseShiftr0Shifted_ = iConfig.getParameter<int>("BaseShiftr0Shifted");
    iConfig_.baseShiftr1Shifted_ = iConfig.getParameter<int>("BaseShiftr1Shifted");
    iConfig_.baseShiftr02_ = iConfig.getParameter<int>("BaseShiftr02");
    iConfig_.baseShiftr12_ = iConfig.getParameter<int>("BaseShiftr12");
    iConfig_.baseShiftchi20_ = iConfig.getParameter<int>("BaseShiftchi20");
    iConfig_.baseShiftchi21_ = iConfig.getParameter<int>("BaseShiftchi21");
    printDebug_ = iConfig.getParameter<bool>("PrintKFDebug");
    const std::string& label = iConfig.getParameter<std::string>("InputLabelKF");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    const std::string& branchTruncated = iConfig.getParameter<std::string>("BranchTruncated");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<tt::StreamsStub>(edm::InputTag(label, branchStubs));
    edGetTokenTracks_ = consumes<tt::StreamsTrack>(edm::InputTag(label, branchTracks));
    edPutTokenStubs_ = produces<tt::StreamsStub>(branchStubs);
    edPutTokenTracks_ = produces<tt::StreamsTrack>(branchTracks);
    edPutTokenNumStatesAccepted_ = produces<int>(branchTracks);
    edPutTokenNumStatesTruncated_ = produces<int>(branchTruncated);
    edPutTokenChi2s_ = produces<std::vector<std::pair<double, double>>>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
    esGetTokenLayerEncoding_ = esConsumes();
  }

  void ProducerKF::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    numRegions_ = setup->numRegions();
    numLayers_ = setup->numLayers();
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    numChannel_ = dataFormats->numChannel(Process::kf);
    kalmanFilterFormats_.beginRun(dataFormats, iConfig_);
  }

  void ProducerKF::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to encode layer
    const LayerEncoding* layerEncoding = &iSetup.getData(esGetTokenLayerEncoding_);
    // empty KF products
    tt::StreamsStub acceptedStubs(numRegions_ * numChannel_ * numLayers_);
    tt::StreamsTrack acceptedTracks(numRegions_ * numChannel_);
    int numStatesAccepted(0);
    int numStatesTruncated(0);
    std::deque<std::pair<double, double>> chi2s;
    // read in SF Product and produce KF product
    const tt::StreamsStub& allStubs = iEvent.get(edGetTokenStubs_);
    const tt::StreamsTrack& allTracks = iEvent.get(edGetTokenTracks_);
    // helper
    auto validFrameT = [](int sum, const tt::FrameTrack& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    auto validFrameS = [](int sum, const tt::FrameStub& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    auto putT = [](const std::vector<TrackKF*>& objects, tt::StreamTrack& stream) {
      auto toFrame = [](TrackKF* object) { return object ? object->frame() : tt::FrameTrack(); };
      stream.reserve(objects.size());
      std::transform(objects.begin(), objects.end(), std::back_inserter(stream), toFrame);
    };
    auto putS = [](const std::vector<StubKF*>& objects, tt::StreamStub& stream) {
      auto toFrame = [](StubKF* object) { return object ? object->frame() : tt::FrameStub(); };
      stream.reserve(objects.size());
      std::transform(objects.begin(), objects.end(), std::back_inserter(stream), toFrame);
    };
    for (int region = 0; region < numRegions_; region++) {
      const int offset = region * numChannel_;
      // count input objects
      int nTracks(0);
      int nStubs(0);
      for (int channel = 0; channel < numChannel_; channel++) {
        const int index = offset + channel;
        const int offsetStubs = index * numLayers_;
        const tt::StreamTrack& tracks = allTracks[index];
        nTracks += std::accumulate(tracks.begin(), tracks.end(), 0, validFrameT);
        for (int layer = 0; layer < numLayers_; layer++) {
          const tt::StreamStub& stubs = allStubs[offsetStubs + layer];
          nStubs += std::accumulate(stubs.begin(), stubs.end(), 0, validFrameS);
        }
      }
      // storage of input data
      std::vector<TrackCTB> tracksCTB;
      tracksCTB.reserve(nTracks);
      std::vector<Stub> stubs;
      stubs.reserve(nStubs);
      // h/w liked organized pointer to input data
      std::vector<std::vector<TrackCTB*>> regionTracks(numChannel_);
      std::vector<std::vector<Stub*>> regionStubs(numChannel_ * numLayers_);
      // read input data
      for (int channel = 0; channel < numChannel_; channel++) {
        const int index = offset + channel;
        const int offsetAll = index * numLayers_;
        const int offsetRegion = channel * numLayers_;
        const tt::StreamTrack& streamTrack = allTracks[index];
        std::vector<TrackCTB*>& tracks = regionTracks[channel];
        tracks.reserve(streamTrack.size());
        for (const tt::FrameTrack& frame : streamTrack) {
          TrackCTB* track = nullptr;
          if (frame.first.isNonnull()) {
            tracksCTB.emplace_back(frame, dataFormats);
            track = &tracksCTB.back();
          }
          tracks.push_back(track);
        }
        for (int layer = 0; layer < numLayers_; layer++) {
          for (const tt::FrameStub& frame : allStubs[offsetAll + layer]) {
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
      std::vector<TrackKF> tracksKF;
      tracksKF.reserve(nTracks);
      std::vector<StubKF> stubsKF;
      stubsKF.reserve(nStubs);
      // object to fit tracks in a processing region
      KalmanFilter kf(setup, dataFormats, layerEncoding, &kalmanFilterFormats_, tracksKF, stubsKF);
      // empty h/w liked organized pointer to output data
      std::vector<std::vector<TrackKF*>> streamsTrack(numChannel_);
      std::vector<std::vector<std::vector<StubKF*>>> streamsStub(numChannel_,
                                                                 std::vector<std::vector<StubKF*>>(numLayers_));
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
