#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackMultiplexer.h"
#include "SimDataFormats/Associations/interface/TTTypes.h"
#include "SimDataFormats/Associations/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Associator.h"

#include <TProfile2D.h>
#include <TH1F.h>

#include <vector>
#include <deque>
#include <map>
#include <set>
#include <cmath>
#include <numeric>
#include <sstream>

namespace trklet {

  /*! \class  trklet::AnalyzerTB
   *  \brief  Class to analyze emulated Track Builder 
   *  \author Thomas Schuh
   *  \date   2025, Nov
   */
  class AnalyzerTB : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
  public:
    AnalyzerTB(const edm::ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override;
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override {}
    void endJob() override {}

  private:
    struct Track {
      Track(const TTTrackRef& ttTrackRef, int region, const std::vector<StubDR*>& seed, const std::vector<StubDR*>& proj)
          : ttTrackRef_(ttTrackRef), region_(region), seed_(seed), proj_(proj) {}
      TTTrackRef ttTrackRef_;
      int region_;
      std::vector<StubDR*> seed_;
      std::vector<StubDR*> proj_;
    };
    // read in tracks and stubs
    void consume(const tt::StreamsTrack&, const tt::StreamsStub&, std::vector<Track>&, std::vector<StubDR>&) const;
    // ED input token of Tracks
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracks_;
    // ED input token of Stubs
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubs_;
    // ED output token for stubs
    edm::EDPutTokenT<tt::StreamsStub> edPutTokenStubs_;
    // ED output token for tracks
    edm::EDPutTokenT<tt::StreamsTrack> edPutTokenTracks_;
    // ED output token for stub association for tracking efficiency
    edm::EDGetTokenT<tt::StubAssociation> edGetTokenEff_;
    // Setup token
    edm::ESGetToken<Setup, trackerDTC::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, trackerDTC::SetupRcd> esGetTokenDataFormats_;
    // Associator token
    edm::ESGetToken<tt::Associator, trackerDTC::SetupRcd> esGetTokenAssociator_;
    // helper class to store configurations
    const Setup* setup_;
    // helper class to read stubs
    const DataFormats* dataFormats_;
    //
    TH1F* hisR_;
    TProfile2D* profR_;
    //
    TH1F* hisPhi_;
    TProfile2D* profPhi_;
    //
    TH1F* hisZ_;
    TProfile2D* profZ_;
  };

  AnalyzerTB::AnalyzerTB(const edm::ParameterSet& iConfig) {
    usesResource("TFileService");
    const std::string& label = iConfig.getParameter<std::string>("InputLabelKF");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    const std::string& labelMC = iConfig.getParameter<std::string>("LabelMC");
    const std::string& branchEff = iConfig.getParameter<std::string>("BranchEff");
    // book in- and output ED products
    edGetTokenTracks_ = consumes(edm::InputTag(label, branchTracks));
    edGetTokenStubs_ = consumes(edm::InputTag(label, branchStubs));
    edGetTokenEff_ = consumes(edm::InputTag(labelMC, branchEff));
    // book ES products
    esGetTokenSetup_ = esConsumes<edm::Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<edm::Transition::BeginRun>();
    esGetTokenAssociator_ = esConsumes();
  }

  void AnalyzerTB::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // book histograms
    edm::Service<TFileService> fs;
    TFileDirectory dir;
    // stub z postion from tracklet residuals plus tracklet seed parameter vs stub z psotion from TTStubs projected to stub radius from tracklet using tracklet seed parameter
    dir = fs->mkdir("Residuals");
    hisR_ = dir.make<TH1F>("His r residual", ";", 128, -.2, .2);
    hisPhi_ = dir.make<TH1F>("His phi residual", ";", 128, -.0002, .0002);
    hisZ_ = dir.make<TH1F>("His z residual", ";", 128, -.5, .5);
    profR_ = dir.make<TProfile2D>("prof r residual", ";", 512, -300, 300., 128, 0., 120.);
    profPhi_ = dir.make<TProfile2D>("prof phi residual", ";", 512, -300, 300., 128, 0., 120.);
    profZ_ = dir.make<TProfile2D>("prof z residual", ";", 512, -300, 300., 128, 0., 120.);
  }

  void AnalyzerTB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    tt::Associator associator = iSetup.getData(esGetTokenAssociator_);
    associator.consume(iEvent.get(edGetTokenEff_));
    // read in TBout Product and produce TM product
    const tt::StreamsStub& streamsStub = iEvent.get(edGetTokenStubs_);
    const tt::StreamsTrack& streamsTrack = iEvent.get(edGetTokenTracks_);
    // storage for tracks and stubs
    std::vector<StubDR> stubs;
    std::vector<Track> tracks;
    // read in tracks and stubs
    consume(streamsTrack, streamsStub, tracks, stubs);
    // analyze in tracks and stubs
    for (const Track& track : tracks) {
      // Do reco-truth matching.
      std::vector<TTStubRef> ttStubRefs;
      ttStubRefs.reserve(setup_->drNumLayers());
      for (StubDR* stub : track.seed_)
        ttStubRefs.push_back(stub->frame().first);
      for (StubDR* stub : track.proj_)
        if (stub)
          ttStubRefs.push_back(stub->frame().first);
      const std::vector<TPPtr>& tpPtrs = associator.associateFinal(ttStubRefs);
      // analyze stub z residuals
      for (StubDR* stub : track.proj_) {
        if (!stub)
          continue;
        const GlobalPoint gp = setup_->stubPosTT(stub->frame().first);
        const double r = gp.perp() - stub->r();
        const double phi = tt::deltaPhi(gp.phi() - stub->phi() - track.region_ * setup_->regRangePhiT());
        const double z = gp.z() - stub->z();
        hisR_->Fill(r);
        hisPhi_->Fill(phi);
        hisZ_->Fill(z);
        profR_->Fill(gp.z(), gp.perp(), std::abs(r));
        profPhi_->Fill(gp.z(), gp.perp(), std::abs(phi));
        profZ_->Fill(gp.z(), gp.perp(), std::abs(z));
      }
    }
  }

  // read in tracks and stubs
  void AnalyzerTB::consume(const tt::StreamsTrack& streamsTrack,
                           const tt::StreamsStub& streamsStub,
                           std::vector<Track>& tracks,
                           std::vector<StubDR>& stubs) const {
    // count tracks and stubs
    int nTracks(0);
    for (const tt::StreamTrack& stream : streamsTrack)
      for (const tt::FrameTrack& frame : stream)
        if (frame.first.isNonnull())
          nTracks++;
    tracks.reserve(nTracks);
    int nStubs(0);
    for (const tt::StreamStub& stream : streamsStub)
      for (const tt::FrameStub& frame : stream)
        if (frame.first.isNonnull())
          nStubs++;
    stubs.reserve(nStubs);
    for (int region = 0; region < setup_->sysNumRegion(); region++) {
      const int offset = region * setup_->drNumLayers();
      const tt::StreamTrack& streamTrack = streamsTrack[region];
      for (int frame = 0; frame < static_cast<int>(streamTrack.size()); frame++) {
        const TTTrackRef& ttTrackRef = streamTrack[frame].first;
        if (ttTrackRef.isNull())
          continue;
        // convert stubs
        std::vector<StubDR*> seed;
        seed.reserve(setup_->tbNumSeedingLayers());
        std::vector<StubDR*> proj(setup_->kfNumLayers(), nullptr);
        for (int layer = 0; layer < setup_->tbNumSeedingLayers(); layer++) {
          stubs.emplace_back(streamsStub[offset + layer][frame], dataFormats_);
          seed.push_back(&stubs.back());
        }
        for (int layer = 0; layer < setup_->kfNumLayers(); layer++) {
          const tt::FrameStub& frameStub = streamsStub[offset + setup_->tbNumSeedingLayers() + layer][frame];
          if (frameStub.first.isNull())
            continue;
          stubs.emplace_back(frameStub, dataFormats_);
          proj[layer] = &stubs.back();
        }
        // create track
        tracks.emplace_back(ttTrackRef, region, seed, proj);
      }
    }
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::AnalyzerTB);
