#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include "L1Trigger/TrackerTFP/interface/Demonstrator.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"

#include <sstream>
#include <utility>
#include <numeric>

namespace trklet {

  /*! \class  trklet::AnalyzerDemonstrator
   *  \brief  calls questasim to simulate the f/w and compares the results with clock-and-bit-accurate emulation.
   *          At the end the number of passing events (not a single bit error) are reported.
   *  \author Thomas Schuh
   *  \date   2022, March
   */
  class AnalyzerDemonstrator : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
  public:
    AnalyzerDemonstrator(const edm::ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override;
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override {}
    void endJob() override;

  private:
    //
    void convert(const edm::Event& iEvent,
                 const edm::EDGetTokenT<tt::StreamsTrack>& tokenTracks,
                 const edm::EDGetTokenT<tt::StreamsStub>& tokenStubs,
                 std::vector<std::vector<tt::Frame>>& bits,
                 bool TB = false) const;
    //
    template <typename T>
    void convert(const T& collection, std::vector<std::vector<tt::Frame>>& bits) const;
    // ED input token of Tracks
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubsIn_;
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubsOut_;
    // ED input token of Stubs
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracksIn_;
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracksOut_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // ChannelAssignment token
    edm::ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
    // Demonstrator token
    edm::ESGetToken<trackerTFP::Demonstrator, tt::SetupRcd> esGetTokenDemonstrator_;
    //
    const tt::Setup* setup_ = nullptr;
    //
    const ChannelAssignment* channelAssignment_ = nullptr;
    //
    const trackerTFP::Demonstrator* demonstrator_ = nullptr;
    //
    int nEvents_ = 0;
    //
    int nEventsSuccessful_ = 0;
    //
    bool TBin_;
    bool TBout_;
  };

  AnalyzerDemonstrator::AnalyzerDemonstrator(const edm::ParameterSet& iConfig) {
    // book in- and output ED products
    const std::string& labelIn = iConfig.getParameter<std::string>("LabelIn");
    const std::string& labelOut = iConfig.getParameter<std::string>("LabelOut");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    edGetTokenStubsIn_ = consumes<tt::StreamsStub>(edm::InputTag(labelIn, branchStubs));
    edGetTokenStubsOut_ = consumes<tt::StreamsStub>(edm::InputTag(labelOut, branchStubs));
    if (labelIn != "ProducerIRin")
      edGetTokenTracksIn_ = consumes<tt::StreamsTrack>(edm::InputTag(labelIn, branchTracks));
    if (labelOut != "ProducerIRin")
      edGetTokenTracksOut_ = consumes<tt::StreamsTrack>(edm::InputTag(labelOut, branchTracks));
    // book ES products
    esGetTokenSetup_ = esConsumes<edm::Transition::BeginRun>();
    esGetTokenChannelAssignment_ = esConsumes<edm::Transition::BeginRun>();
    esGetTokenDemonstrator_ = esConsumes<edm::Transition::BeginRun>();
    //
    TBin_ = labelIn == "l1tTTTracksFromTrackletEmulation";
    TBout_ = labelOut == "l1tTTTracksFromTrackletEmulation";
  }

  void AnalyzerDemonstrator::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    //
    setup_ = &iSetup.getData(esGetTokenSetup_);
    //
    channelAssignment_ = &iSetup.getData(esGetTokenChannelAssignment_);
    //
    demonstrator_ = &iSetup.getData(esGetTokenDemonstrator_);
  }

  void AnalyzerDemonstrator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    nEvents_++;
    std::vector<std::vector<tt::Frame>> input;
    std::vector<std::vector<tt::Frame>> output;
    convert(iEvent, edGetTokenTracksIn_, edGetTokenStubsIn_, input, TBin_);
    convert(iEvent, edGetTokenTracksOut_, edGetTokenStubsOut_, output, TBout_);
    if (demonstrator_->analyze(input, output))
      nEventsSuccessful_++;
  }

  //
  void AnalyzerDemonstrator::convert(const edm::Event& iEvent,
                                     const edm::EDGetTokenT<tt::StreamsTrack>& tokenTracks,
                                     const edm::EDGetTokenT<tt::StreamsStub>& tokenStubs,
                                     std::vector<std::vector<tt::Frame>>& bits,
                                     bool TB) const {
    const bool tracks = !tokenTracks.isUninitialized();
    const bool stubs = !tokenStubs.isUninitialized();
    edm::Handle<tt::StreamsStub> handleStubs;
    edm::Handle<tt::StreamsTrack> handleTracks;
    int numChannelStubs(0);
    if (stubs) {
      iEvent.getByToken<tt::StreamsStub>(tokenStubs, handleStubs);
      numChannelStubs = handleStubs->size();
    }
    int numChannelTracks(0);
    if (tracks) {
      iEvent.getByToken<tt::StreamsTrack>(tokenTracks, handleTracks);
      numChannelTracks = handleTracks->size();
    }
    numChannelTracks /= setup_->numRegions();
    numChannelStubs /= (setup_->numRegions() * (tracks ? numChannelTracks : 1));
    if (TB)
      numChannelStubs = channelAssignment_->numChannelsStub();
    bits.reserve(numChannelTracks + numChannelStubs);
    for (int region = 0; region < setup_->numRegions(); region++) {
      if (tracks) {
        const int offsetTracks = region * numChannelTracks;
        for (int channelTracks = 0; channelTracks < numChannelTracks; channelTracks++) {
          int offsetStubs = (region * numChannelTracks + channelTracks) * numChannelStubs;
          if (TB) {
            numChannelStubs =
                channelAssignment_->numProjectionLayers(channelTracks) + channelAssignment_->numSeedingLayers();
            offsetStubs = channelAssignment_->offsetStub(offsetTracks + channelTracks);
          }
          if (tracks)
            convert(handleTracks->at(offsetTracks + channelTracks), bits);
          if (stubs) {
            for (int channelStubs = 0; channelStubs < numChannelStubs; channelStubs++)
              convert(handleStubs->at(offsetStubs + channelStubs), bits);
          }
        }
      } else {
        const int offsetStubs = region * numChannelStubs;
        for (int channelStubs = 0; channelStubs < numChannelStubs; channelStubs++)
          convert(handleStubs->at(offsetStubs + channelStubs), bits);
      }
    }
  }

  //
  template <typename T>
  void AnalyzerDemonstrator::convert(const T& collection, std::vector<std::vector<tt::Frame>>& bits) const {
    bits.emplace_back();
    std::vector<tt::Frame>& bvs = bits.back();
    bvs.reserve(collection.size());
    transform(collection.begin(), collection.end(), back_inserter(bvs), [](const auto& frame) { return frame.second; });
  }

  void AnalyzerDemonstrator::endJob() {
    std::stringstream log;
    log << "Successrate: " << nEventsSuccessful_ << " / " << nEvents_ << " = " << nEventsSuccessful_ / (double)nEvents_;
    edm::LogPrint(moduleDescription().moduleName()) << log.str();
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::AnalyzerDemonstrator);
