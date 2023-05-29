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

using namespace std;
using namespace edm;
using namespace tt;
using namespace trackerTFP;

namespace trklet {

  /*! \class  trklet::AnalyzerDemonstrator
   *  \brief  Class to demontrate correctness of track trigger emulators
   *          by comparing FW with SW
   *  \author Thomas Schuh
   *  \date   2022, March
   */
  class AnalyzerDemonstrator : public one::EDAnalyzer<one::WatchRuns> {
  public:
    AnalyzerDemonstrator(const ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const Run& iEvent, const EventSetup& iSetup) override;
    void analyze(const Event& iEvent, const EventSetup& iSetup) override;
    void endRun(const Run& iEvent, const EventSetup& iSetup) override {}
    void endJob() override;

  private:
    //
    void convert(const Event& iEvent,
                 const EDGetTokenT<StreamsTrack>& tokenTracks,
                 const EDGetTokenT<StreamsStub>& tokenStubs,
                 vector<vector<Frame>>& bits) const;
    //
    template <typename T>
    void convert(const T& collection, vector<vector<Frame>>& bits) const;
    // ED input token of Tracks
    EDGetTokenT<StreamsStub> edGetTokenStubsIn_;
    EDGetTokenT<StreamsStub> edGetTokenStubsOut_;
    // ED input token of Stubs
    EDGetTokenT<StreamsTrack> edGetTokenTracksIn_;
    EDGetTokenT<StreamsTrack> edGetTokenTracksOut_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // ChannelAssignment token
    ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
    // Demonstrator token
    ESGetToken<Demonstrator, DemonstratorRcd> esGetTokenDemonstrator_;
    //
    const Setup* setup_ = nullptr;
    //
    const ChannelAssignment* channelAssignment_ = nullptr;
    //
    const Demonstrator* demonstrator_ = nullptr;
    //
    int nEvents_ = 0;
    //
    int nEventsSuccessful_ = 0;
  };

  AnalyzerDemonstrator::AnalyzerDemonstrator(const ParameterSet& iConfig) {
    // book in- and output ED products
    const string& labelIn = iConfig.getParameter<string>("LabelIn");
    const string& labelOut = iConfig.getParameter<string>("LabelOut");
    const string& branchStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchAcceptedTracks");
    edGetTokenStubsIn_ = consumes<StreamsStub>(InputTag(labelIn, branchStubs));
    edGetTokenStubsOut_ = consumes<StreamsStub>(InputTag(labelOut, branchStubs));
    if (labelOut != "TrackFindingTrackletProducerKFout")
      edGetTokenStubsOut_ = consumes<StreamsStub>(InputTag(labelOut, branchStubs));
    if (labelIn != "TrackFindingTrackletProducerIRin")
      edGetTokenTracksIn_ = consumes<StreamsTrack>(InputTag(labelIn, branchTracks));
    if (labelOut != "TrackFindingTrackletProducerIRin")
      edGetTokenTracksOut_ = consumes<StreamsTrack>(InputTag(labelOut, branchTracks));
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenChannelAssignment_ = esConsumes<ChannelAssignment, ChannelAssignmentRcd, Transition::BeginRun>();
    esGetTokenDemonstrator_ = esConsumes<Demonstrator, DemonstratorRcd, Transition::BeginRun>();
  }

  void AnalyzerDemonstrator::beginRun(const Run& iEvent, const EventSetup& iSetup) {
    //
    setup_ = &iSetup.getData(esGetTokenSetup_);
    //
    channelAssignment_ = &iSetup.getData(esGetTokenChannelAssignment_);
    //
    demonstrator_ = &iSetup.getData(esGetTokenDemonstrator_);
  }

  void AnalyzerDemonstrator::analyze(const Event& iEvent, const EventSetup& iSetup) {
    nEvents_++;
    vector<vector<Frame>> input;
    vector<vector<Frame>> output;
    convert(iEvent, edGetTokenTracksIn_, edGetTokenStubsIn_, input);
    convert(iEvent, edGetTokenTracksOut_, edGetTokenStubsOut_, output);
    if (demonstrator_->analyze(input, output))
      nEventsSuccessful_++;
  }

  //
  void AnalyzerDemonstrator::convert(const Event& iEvent,
                                     const EDGetTokenT<StreamsTrack>& tokenTracks,
                                     const EDGetTokenT<StreamsStub>& tokenStubs,
                                     vector<vector<Frame>>& bits) const {
    const bool tracks = !tokenTracks.isUninitialized();
    const bool stubs = !tokenStubs.isUninitialized();
    Handle<StreamsStub> handleStubs;
    Handle<StreamsTrack> handleTracks;
    int numChannelTracks(0);
    if (tracks) {
      iEvent.getByToken<StreamsTrack>(tokenTracks, handleTracks);
      numChannelTracks = handleTracks->size();
    }
    numChannelTracks /= setup_->numRegions();
    int numChannelStubs(0);
    vector<int> numChannelsStubs(numChannelTracks, 0);
    if (stubs) {
      iEvent.getByToken<StreamsStub>(tokenStubs, handleStubs);
      numChannelStubs = handleStubs->size() / setup_->numRegions();
      const int numChannel = tracks ? numChannelStubs / numChannelTracks : numChannelStubs;
      numChannelsStubs = vector<int>(numChannelTracks, numChannel);
    }
    bits.reserve(numChannelTracks + numChannelStubs);
    for (int region = 0; region < setup_->numRegions(); region++) {
      if (tracks) {
        const int offsetTracks = region * numChannelTracks;
        for (int channelTracks = 0; channelTracks < numChannelTracks; channelTracks++) {
          const int offsetStubs =
              region * numChannelStubs +
              accumulate(numChannelsStubs.begin(), next(numChannelsStubs.begin(), channelTracks), 0);
          convert(handleTracks->at(offsetTracks + channelTracks), bits);
          if (stubs) {
            for (int channelStubs = 0; channelStubs < numChannelsStubs[channelTracks]; channelStubs++)
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
  void AnalyzerDemonstrator::convert(const T& collection, vector<vector<Frame>>& bits) const {
    bits.emplace_back();
    vector<Frame>& bvs = bits.back();
    bvs.reserve(collection.size());
    transform(collection.begin(), collection.end(), back_inserter(bvs), [](const auto& frame) { return frame.second; });
  }

  void AnalyzerDemonstrator::endJob() {
    stringstream log;
    log << "Successrate: " << nEventsSuccessful_ << " / " << nEvents_ << " = " << nEventsSuccessful_ / (double)nEvents_;
    LogPrint("L1Trigger/TrackerTFP") << log.str();
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::AnalyzerDemonstrator);