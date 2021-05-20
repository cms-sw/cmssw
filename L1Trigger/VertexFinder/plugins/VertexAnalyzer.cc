#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "L1Trigger/VertexFinder/interface/AnalysisSettings.h"
#include "L1Trigger/VertexFinder/interface/InputData.h"
#include "L1Trigger/VertexFinder/interface/L1TrackTruthMatched.h"
#include "L1Trigger/VertexFinder/interface/RecoVertex.h"
#include "L1Trigger/VertexFinder/interface/selection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"

#include "TEfficiency.h"
#include "TGraphAsymmErrors.h"
#include "TGraphErrors.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TPad.h"
#include "TProfile.h"

#include <map>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

namespace l1tVertexFinder {

  class VertexAnalyzer : public edm::EDAnalyzer {
  public:
    explicit VertexAnalyzer(const edm::ParameterSet&);
    ~VertexAnalyzer() override;

  private:
    template <typename T>
    int getIndex(std::vector<T> collection, T reference);
    void beginJob() override;
    void analyze(const edm::Event& evt, const edm::EventSetup& setup) override;
    void endJob() override;

    // define types for stub-related classes
    typedef edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>> DetSetVec;
    typedef TTTrackAssociationMap<Ref_Phase2TrackerDigi_> TTTrackAssMap;
    typedef TTStubAssociationMap<Ref_Phase2TrackerDigi_> TTStubAssMap;
    typedef TTClusterAssociationMap<Ref_Phase2TrackerDigi_> TTClusterAssMap;
    typedef edm::View<TTTrack<Ref_Phase2TrackerDigi_>> TTTrackCollectionView;

    // references to tags containing information relevant to perofrmance analysis
    const edm::EDGetTokenT<edm::HepMCProduct> hepMCInputTag;
    const edm::EDGetTokenT<edm::View<reco::GenParticle>> genParticleInputTag;
    const edm::EDGetTokenT<TrackingParticleCollection> tpInputTag;
    const edm::EDGetTokenT<TTTrackAssMap> trackTruthInputTag;
    const edm::EDGetTokenT<DetSetVec> stubInputTag;
    const edm::EDGetTokenT<TTStubAssMap> stubTruthInputTag;
    const edm::EDGetTokenT<TTClusterAssMap> clusterTruthInputTag;
    const edm::EDGetTokenT<TTTrackCollectionView> l1TracksToken_;
    const edm::EDGetTokenT<std::vector<l1t::Vertex>> l1VerticesToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;

    const bool printResults_;

    // storage class for configuration parameters
    AnalysisSettings settings_;

    //edm::Service<TFileService> fs_;
    // Histograms for Vertex Reconstruction

    TH1F* hisNoRecoVertices_;
    TH1F* hisNoPileUpVertices_;
    TH1F* hisRecoVertexZ0Resolution_;
    TH1F* hisRecoVertexZ0Separation_;
    TH1F* hisRecoVertexPTResolution_;
    TH2F* hisNoRecoVsNoTruePileUpVertices_;
    TH2F* hisNoTracksFromPrimaryVertex_;
    TProfile* hisRecoVertexPTResolutionVsTruePt_;
    TH2F* hisNoTrueTracksFromPrimaryVertex_;
    TH1F* hisRecoPrimaryVertexZ0width_;
    TH1F* hisRecoPileUpVertexZ0width_;
    TH1F* hisRecoVertexZ0Spacing_;
    TH1F* hisPrimaryVertexZ0width_;
    TH1F* hisPileUpVertexZ0_;
    TH1F* hisPileUpVertexZ0width_;
    TH1F* hisPileUpVertexZ0Spacing_;
    TH1F* hisRecoPileUpVertexZ0resolution_;
    TH1F* hisRatioMatchedTracksInPV_;
    TH1F* hisFakeTracksRateInPV_;
    TH1F* hisTrueTracksRateInPV_;
    TH2F* hisRecoVertexPTVsTruePt_;
    TH1F* hisUnmatchZ0distance_;
    TH1F* hisUnmatchZ0MinDistance_;
    TH1F* hisUnmatchPt_;
    TH1F* hisUnmatchEta_;
    TH1F* hisUnmatchTruePt_;
    TH1F* hisUnmatchTrueEta_;
    TH1F* hisLostPVtracks_;
    TH1F* hisUnmatchedPVtracks_;
    TH1F* hisNumVxIterations_;
    TH1F* hisNumVxIterationsPerTrack_;
    TH1F* hisCorrelatorInputTracks_;
    TH1F* hisCorrelatorTPInputTracks_;
    TH1F* hisCorrelatorInputVertices_;
    TH1F* hisCorrelatorTPInputVertices_;

    TH1F* hisRecoPrimaryVertexVsTrueZ0_;
    TH1F* hisPrimaryVertexTrueZ0_;
    TH1F* hisRecoVertexPT_;
    TH1F* hisRecoPileUpVertexPT_;
    TH1F* hisRecoVertexOffPT_;
    TH1F* hisRecoVertexTrackRank_;

    TProfile* hisRecoPrimaryVertexResolutionVsTrueZ0_;

    TH1F* hisUnmatchedVertexZ0distance_;

    TH1F* hisGenVertexPt_;
    TH1F* hisGenTkVertexPt_;

    TH1F* hisGenVertexTrackPt_;
    TH1F* hisGenVertexNumTracks_;

    TH1F* hisRecoVertexVsNumGenTracks_;
    TH1F* hisRecoVertexVsGenVertexPt_;
    TH1F* hisRecoGenuineVertexVsGenTkVertexPt_;

    TH1F* hisRecoVertexVsGenTkVertexPtForEff_;

    TH1F* hisPUVertexPt_;
    TH1F* hisPUTkVertexPt_;
    TH1F* hisPUVertexTrackPt_;
    TH1F* hisPUVertexNumTracks_;

    std::vector<TH1F*> hisPTevents_;

    TEfficiency* PVefficiencyVsTrueZ0_;
  };

  VertexAnalyzer::VertexAnalyzer(const edm::ParameterSet& iConfig)
      : hepMCInputTag(consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("hepMCInputTag"))),
        genParticleInputTag(
            consumes<edm::View<reco::GenParticle>>(iConfig.getParameter<edm::InputTag>("genParticleInputTag"))),
        tpInputTag(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("tpInputTag"))),
        trackTruthInputTag(consumes<TTTrackAssMap>(iConfig.getParameter<edm::InputTag>("l1TracksTruthMapInputTags"))),
        stubInputTag(consumes<DetSetVec>(iConfig.getParameter<edm::InputTag>("stubInputTag"))),
        stubTruthInputTag(consumes<TTStubAssMap>(iConfig.getParameter<edm::InputTag>("stubTruthInputTag"))),
        clusterTruthInputTag(consumes<TTClusterAssMap>(iConfig.getParameter<edm::InputTag>("clusterTruthInputTag"))),
        l1TracksToken_(consumes<TTTrackCollectionView>(iConfig.getParameter<edm::InputTag>("l1TracksInputTag"))),
        l1VerticesToken_(consumes<std::vector<l1t::Vertex>>(iConfig.getParameter<edm::InputTag>("l1VerticesInputTag"))),
        trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>(edm::ESInputTag("", ""))),
        trackerTopologyToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>(edm::ESInputTag("", ""))),
        printResults_(iConfig.getParameter<bool>("printResults")),
        settings_(iConfig) {
    // Configure TH1 for plotting
    TH1::SetDefaultSumw2(true);

    // create performance histograms
    edm::Service<TFileService> fs_;
    TFileDirectory inputDir = fs_->mkdir("VertexReconstruction");

    // distributions of the generator level information
    hisGenVertexPt_ = inputDir.make<TH1F>("hisGenVertexPt_", "hisGenVertexPt_", 100, 0, 500);
    hisGenTkVertexPt_ = inputDir.make<TH1F>("hisGenTkVertexPt_", "hisGenTkVertexPt_", 100, 0, 500);
    hisGenVertexTrackPt_ = inputDir.make<TH1F>("hisGenVertexTrackPt_", "hisGenVertexTrackPt_", 50, 0, 300);
    hisGenVertexNumTracks_ = inputDir.make<TH1F>("hisGenVertexNumTracks_", "hisGenVertexNumTracks_", 20, 0, 50);

    // information about pile-up vertices
    hisPUVertexPt_ = inputDir.make<TH1F>("hisPUVertexPt_", "hisPUVertexPt_", 50, 0, 300);
    hisPUVertexTrackPt_ = inputDir.make<TH1F>("hisPUVertexTrackPt_", "hisPUVertexTrackPt_", 50, 0, 300);
    hisPUVertexNumTracks_ = inputDir.make<TH1F>("hisPUVertexNumTracks_", "hisPUVertexNumTracks_", 20, 0, 50);

    hisRecoGenuineVertexVsGenTkVertexPt_ = inputDir.make<TH1F>(
        "hisRecoGenuineVertexVsGenTkVertexPt_", "hisRecoGenuineVertexVsGenTkVertexPt_", 100, 0, 500);
    hisRecoVertexVsGenVertexPt_ =
        inputDir.make<TH1F>("hisRecoVertexVsGenVertexPt_", "hisRecoVertexVsGenVertexPt_", 100, 0, 500);
    hisRecoVertexVsGenTkVertexPtForEff_ =
        inputDir.make<TH1F>("hisRecoVertexVsGenTkVertexPtForEff_", "hisRecoVertexVsGenTkVertexPtForEff_", 100, 0, 500);
    hisRecoVertexVsNumGenTracks_ =
        inputDir.make<TH1F>("hisRecoVertexVsNumGenTracks_", "hisRecoVertexVsNumGenTracks_", 20, 0, 50);

    hisNoRecoVertices_ =
        inputDir.make<TH1F>("hisNoRecoVertices_", "No. reconstructed Vertices; No. reco vertices; Events", 50, 0, 50);
    hisNoPileUpVertices_ =
        inputDir.make<TH1F>("hisNoPileUpVertices_", "No. pile-up Vertices; No. pile-up vertices; Events", 50, 0, 50);
    hisNoRecoVsNoTruePileUpVertices_ =
        inputDir.make<TH2F>("hisNoRecoVsNoTruePileUpVertices_",
                            "No. reconstructed pile-up vertices vs. no. true pile-up vertices; No. reco pile-up "
                            "vertices; No. true pile-up vertices",
                            50,
                            0,
                            50,
                            50,
                            0,
                            50);
    hisRecoVertexZ0Resolution_ =
        inputDir.make<TH1F>("hisRecoVertexZ0Resolution",
                            "Reconstructed primary vertex z_{0} resolution; z_{0} Resolution [cm]; Counts",
                            200,
                            -1.,
                            1.);
    hisRecoVertexZ0Separation_ =
        inputDir.make<TH1F>("hisRecoVertexZ0Separation",
                            "Separation between reconstructed and generator primary vertex in z_{0}; z_{0} Separation "
                            "(|z_{0}^{gen} - z_{0}^{reco}| [cm]; Counts",
                            100,
                            0.,
                            1.);
    hisRecoVertexPTResolution_ =
        inputDir.make<TH1F>("hisRecoVertexPTResolution",
                            "Reconstructed primary vertex p_{T} relative resolution; p_{T} relative Resolution; Counts",
                            100,
                            0,
                            1.);
    hisRecoVertexPTResolutionVsTruePt_ = inputDir.make<TProfile>(
        "hisRecoVertexPTResolutionVsTruePt",
        "Reconstructed primary vertex relative p_{T} resolution vs. True Pt; True p_{T}; p_{T} Resolution [GeV]",
        100,
        0,
        500);
    hisRecoVertexPTVsTruePt_ =
        inputDir.make<TH2F>("hisRecoVertexPtVsTruePt_",
                            "Reconstructed primary vertex p_{T}  vs. True Pt; p_{T} [GeV]; True p_{T}",
                            100,
                            0,
                            500.,
                            100,
                            0,
                            500.);
    hisNoTracksFromPrimaryVertex_ = inputDir.make<TH2F>(
        "hisNoTracksFromPrimaryVertex_",
        "No. of Tracks from Primary Vertex (Reco vs. Truth); no. Reco Tracks in PV; no. Truth Tracks in PV ",
        50,
        0,
        50,
        50,
        0,
        50);
    hisNoTrueTracksFromPrimaryVertex_ = inputDir.make<TH2F>(
        "hisNoTrueTracksFromPrimaryVertex_",
        "No. of Matched Tracks from Primary Vertex (Reco vs. Truth); no. Reco Tracks in PV; no. Truth Tracks in PV ",
        50,
        0,
        50,
        50,
        0,
        50);
    hisRecoPrimaryVertexZ0width_ =
        inputDir.make<TH1F>("hisRecoPrimaryVertexZ0width_", "Reconstructed primary vertex z_{0} width", 100, 0, 0.5);
    hisRatioMatchedTracksInPV_ =
        inputDir.make<TH1F>("hisRatioMatchedTracksInPV", "Primary vertex matching ratio ", 20, 0, 1.);
    hisFakeTracksRateInPV_ = inputDir.make<TH1F>(
        "hisFakeTracksRateInPV", "Percentage of fake tracks in reconstructed primary vertex", 20, 0, 1.);
    hisTrueTracksRateInPV_ = inputDir.make<TH1F>(
        "hisTrueTracksRateInPV", "Percentage of true tracks in reconstructed primary vertex", 20, 0, 1.);

    hisRecoPrimaryVertexVsTrueZ0_ =
        inputDir.make<TH1F>("hisRecoPrimaryVertexVsTrueZ0_",
                            "No. of reconstructed primary vertices per true z_{0}; true z_{0} [cm]; No. Vertices",
                            50,
                            -25.,
                            25.);
    hisRecoPrimaryVertexResolutionVsTrueZ0_ = inputDir.make<TProfile>(
        "hisRecoPrimaryVertexResolutionVsTrueZ0_",
        "No. of reconstructed primary vertices per true z_{0}; true z_{0} [cm]; Resolution [mm]",
        100,
        -25.,
        25.);
    hisRecoVertexPT_ = inputDir.make<TH1F>("hisRecoVertexPT_", "; #Sigma_{vtx} p_{T} (GeV) ; Entries", 100, 0, 500);
    hisRecoPileUpVertexPT_ =
        inputDir.make<TH1F>("hisRecoPileUpVertexPT_", "; #Sigma_{vtx} p_{T} (GeV) ; Entries", 50, 0, 200);
    hisRecoVertexOffPT_ =
        inputDir.make<TH1F>("hisRecoVertexOffPT_", "; #Sigma_{vtx} p_{T} (GeV) ; Entries", 50, 0, 200);
    hisRecoVertexTrackRank_ = inputDir.make<TH1F>("hisRecoVertexTrackRank_", "; Track Rank; Entries", 20, 0, 20);

    // Plot number of reconstructed vertices against number of true vertices

    // *** Vertex Reconstruction algorithm Plots ***
    hisUnmatchedVertexZ0distance_ = inputDir.make<TH1F>(
        "hisUnmatchedVertexZ0distance_",
        " Unmatched primary vertex z_{0} - true PV z_{0}; |z_{0}^{reco} - z_{0}^{true}| [cm]; Counts",
        200,
        1.,
        5.);
    hisUnmatchZ0distance_ = inputDir.make<TH1F>("hisUnmatchZ0distance_",
                                                "z0 distance from reconstructed privary vertex of unmatched tracks; "
                                                "|z_{0}^{track} - z_{0}^{vertex}|; no. L1 Tracks",
                                                100,
                                                0,
                                                5.);
    hisUnmatchZ0MinDistance_ =
        inputDir.make<TH1F>("hisUnmatchZ0MinDistance_",
                            "z0 distance from the closest track in reconstructed privary vertex of unmatched tracks; "
                            "|z_{0}^{track} - z_{0}^{PV track}|; no. L1 Tracks",
                            100,
                            0,
                            5.);
    hisUnmatchPt_ = inputDir.make<TH1F>(
        "hisUnmatchPt_", "Transverse momentum of unmatched PV tracks; p_{T} [GeV/c]; no. L1 Tracks", 100, 0, 100.);
    hisUnmatchEta_ =
        inputDir.make<TH1F>("hisUnmatchEta_", "#eta of unmatched PV tracks; #eta; no. L1 Tracks", 100, 2.4, 2.4);
    hisUnmatchTruePt_ =
        inputDir.make<TH1F>("hisUnmatchTruePt_",
                            "True transverse momentum of unmatched PV tracks; p_{T} [GeV/c]; no. L1 Tracks",
                            100,
                            0,
                            100.);
    hisUnmatchTrueEta_ = inputDir.make<TH1F>(
        "hisUnmatchTrueEta_", "True #eta of unmatched PV tracks; #eta; no. L1 Tracks", 100, 2.4, 2.4);
    hisUnmatchedPVtracks_ = inputDir.make<TH1F>(
        "hisUnmatchedPVtracks_",
        " No. of tracks from primary collision that are misassigned; No. misassigned Tracks, No. Events ",
        100,
        0,
        100);

    // ** PileUp vertices plot **
    hisRecoPileUpVertexZ0width_ =
        inputDir.make<TH1F>("hisRecoPileUPVertexZ0width_", "Reconstructed pile-up vertex z_{0} width", 100, 0, 1.);
    hisRecoPileUpVertexZ0resolution_ =
        inputDir.make<TH1F>("hisRecoPileUPVertexZ0resolution_",
                            "Reconstructed pile-up vertex z_{0} resolution; #sigma_{z_{0}} [cm]",
                            100,
                            0,
                            1.);

    hisRecoVertexZ0Spacing_ =
        inputDir.make<TH1F>("hisRecoVertexZ0Spacing", "Reconstructed intravertex z_{0} distance", 100, 0., 5.);

    hisPrimaryVertexTrueZ0_ =
        inputDir.make<TH1F>("hisPrimaryVertexTrueZ0_",
                            "No. of gen primary vertices per true z_{0}; true z_{0} [cm]; No. Vertices",
                            50,
                            -25.,
                            25.);

    hisPrimaryVertexZ0width_ =
        inputDir.make<TH1F>("hisPrimaryVertexZ0width_", "Primary vertex z_{0} width", 100, 0, 0.5);
    hisPileUpVertexZ0_ = inputDir.make<TH1F>("hisPileUpVertexZ0", "Pile Up vertex z_{0} position", 200, -15., 15.);
    hisPileUpVertexZ0Spacing_ =
        inputDir.make<TH1F>("hisPileUpVertexZ0Spacing", "Pile Up intravertex z_{0} distance", 100, 0., 5.);
    hisPileUpVertexZ0width_ = inputDir.make<TH1F>("hisPileUpVertexZ0width_", "Pile Up vertex z_{0} width", 100, 0, 0.5);

    hisLostPVtracks_ = inputDir.make<TH1F>(
        "hisLostPVtracks_",
        " No. of tracks from primary collision that are not found by the L1 Track Finder; No. Lost Tracks, No. Events ",
        100,
        0,
        100);

    hisNumVxIterations_ = inputDir.make<TH1F>(
        "hisNumVxIterations_", "Number of Iterations (Vertex Reconstruction); No. Iterations; Entries", 100, 0, 500);
    hisNumVxIterationsPerTrack_ =
        inputDir.make<TH1F>("hisNumVxIterationsPerTrack_",
                            "Number of Iterations per Track(Vertex Reconstruction); No. Iterations; Entries",
                            100,
                            0,
                            200);

    hisCorrelatorInputTracks_ = inputDir.make<TH1F>(
        "hisCorrelatorInputTracks_", "Number of Input Tracks at L1 correlator; No. L1 Tracks; Entries", 30, 0, 200);
    hisCorrelatorTPInputTracks_ = inputDir.make<TH1F>(
        "hisCorrelatorTPInputTracks_", "Number of Input Tracks at L1 correlator; No. L1 Tracks; Entries", 30, 0, 200);

    hisCorrelatorInputVertices_ =
        inputDir.make<TH1F>("hisCorrelatorInputVertices_",
                            "Number of Input Vertices at L1 correlator; No. L1TkVertices; Entries",
                            25,
                            0,
                            100);
    hisCorrelatorTPInputVertices_ =
        inputDir.make<TH1F>("hisCorrelatorTPInputVertices_",
                            "Number of Input Vertices at L1 correlator; No. L1TKVertices; Entries",
                            25,
                            0,
                            100);

    std::vector<unsigned int> emptyVector;
    emptyVector.assign(10, 0);
  }

  template <typename T>
  int VertexAnalyzer::getIndex(std::vector<T> collection, const T reference) {
    int index = -1;
    auto itr = std::find(collection.begin(), collection.end(), reference);
    if (itr != collection.end()) {
      index = std::distance(collection.begin(), itr);
    }
    return index;
  }

  void VertexAnalyzer::beginJob(){};

  void VertexAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // Note useful info about MC truth particles and about reconstructed stubs .
    InputData inputData(iEvent,
                        iSetup,
                        settings_,
                        hepMCInputTag,
                        genParticleInputTag,
                        trackerGeometryToken_,
                        trackerTopologyToken_,
                        tpInputTag,
                        stubInputTag,
                        stubTruthInputTag,
                        clusterTruthInputTag);

    edm::Handle<TTTrackCollectionView> l1TracksHandle;
    iEvent.getByToken(l1TracksToken_, l1TracksHandle);

    std::vector<L1TrackTruthMatched> l1Tracks;
    l1Tracks.reserve(l1TracksHandle->size());
    {
      edm::Handle<TTTrackAssMap> mcTruthTTTrackHandle;
      edm::Handle<TTStubAssMap> mcTruthTTStubHandle;
      edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle;
      iEvent.getByToken(trackTruthInputTag, mcTruthTTTrackHandle);
      iEvent.getByToken(stubTruthInputTag, mcTruthTTStubHandle);
      iEvent.getByToken(clusterTruthInputTag, mcTruthTTClusterHandle);

      for (const auto& track : l1TracksHandle->ptrs())
        l1Tracks.push_back(L1TrackTruthMatched(track, inputData.getTPTranslationMap(), mcTruthTTTrackHandle));
    }

    std::vector<const L1TrackTruthMatched*> l1TrackPtrs;
    l1TrackPtrs.reserve(l1Tracks.size());

    // TODO: REVIEW. This check on tracks should ideally be done with an edm
    // filter so as to be shared between the producer and analyser. The current
    // implementation duplicates the requirements and can potentially bias the
    // analysis results.
    for (const auto& track : l1Tracks) {
      if (track.pt() > settings_.vx_TrackMinPt()) {
        if (track.pt() < 50 or track.getNumStubs() > 5) {
          l1TrackPtrs.push_back(&track);
        }
      }
    }

    edm::Handle<std::vector<l1t::Vertex>> l1VerticesHandle;
    iEvent.getByToken(l1VerticesToken_, l1VerticesHandle);

    // extract VF parameters
    unsigned int numInputTracks = l1TrackPtrs.size();
    unsigned int numVertices = l1VerticesHandle->size();

    if (settings_.debug() > 0)
      edm::LogInfo("VertexAnalyzer") << "analyzer::Input Tracks to L1 Correlator " << numInputTracks;

    const size_t primaryVertexIndex =
        (l1VerticesHandle->empty() ? 0
                                   : &l1tVertexFinder::getPrimaryVertex(*l1VerticesHandle) - &l1VerticesHandle->at(0));

    std::vector<RecoVertexWithTP*> recoVertices;
    recoVertices.reserve(numVertices);

    const Vertex& TruePrimaryVertex = inputData.getPrimaryVertex();

    // create a map for associating fat reco tracks with their underlying
    // TTTrack pointers
    std::map<const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>, const L1TrackTruthMatched*> trackAssociationMap;

    // get a list of reconstructed tracks with references to their TPs
    for (const auto& trackIt : l1Tracks) {
      trackAssociationMap.insert(std::pair<const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>, const L1TrackTruthMatched*>(
          trackIt.getTTTrackPtr(), &trackIt));
    }

    // generate reconstructed vertices (starting at 1 avoids PV)
    for (unsigned int i = 0; i < numVertices; ++i) {
      RecoVertexWithTP* recoVertex = new RecoVertexWithTP(l1VerticesHandle->at(i), trackAssociationMap);
      recoVertex->computeParameters(
          settings_.vx_weightedmean(), settings_.vx_TrackMaxPt(), settings_.vx_TrackMaxPtBehavior());
      if (settings_.vx_algo() == Algorithm::Kmeans || settings_.vx_algo() == Algorithm::HPV)
        recoVertex->setZ(l1VerticesHandle->at(i).z0());
      /*
      RecoVertex<> recoVertexBase = RecoVertex<>();

      // populate vertex with tracks
      for (const auto& track : l1VerticesHandle->at(i).tracks()) {
        recoVertexBase.insert(new L1Track(track));
      }

      recoVertexBase.setZ(l1VerticesHandle->at(i).z0());
      RecoVertexWithTP* recoVertex = new RecoVertexWithTP(recoVertexBase, trackAssociationMap);
      recoVertex->computeParameters(settings_.vx_weightedmean());
      if (settings_.vx_algo() == Algorithm::Kmeans || settings_.vx_algo() == Algorithm::HPV ||
          settings_.vx_algo() == Algorithm::FastHisto)
        recoVertex->setZ(recoVertexBase.z0());
    */
      recoVertices.emplace_back(recoVertex);
    }

    // Associate true primary vertex with the closest reconstructed vertex
    unique_ptr<RecoVertexWithTP> RecoPrimaryVertex(l1VerticesHandle->empty() ? new RecoVertexWithTP(-9999.)
                                                                             : recoVertices.at(primaryVertexIndex));

    if (settings_.debug() > 2 and numVertices > 0) {
      edm::LogInfo("VertexAnalyzer") << "analyzer::Num Found Vertices " << numVertices
                                     << "\nReconstructed Primary Vertex z0 " << RecoPrimaryVertex->z0() << " pT "
                                     << RecoPrimaryVertex->pT();
    }

    hisGenVertexPt_->Fill(inputData.genPt());
    hisGenTkVertexPt_->Fill(TruePrimaryVertex.pT());
    hisGenVertexNumTracks_->Fill(TruePrimaryVertex.numTracks());

    for (const TP& tp : TruePrimaryVertex.tracks()) {
      hisGenVertexTrackPt_->Fill(tp->pt());
    }

    for (const Vertex& vertex : inputData.getPileUpVertices()) {
      hisPUVertexPt_->Fill(vertex.pT());
      hisPUVertexNumTracks_->Fill(vertex.numTracks());
      for (const TP& tp : vertex.tracks()) {
        hisPUVertexTrackPt_->Fill(tp->pt());
      }
    }

    if (RecoPrimaryVertex->pT() > 100.) {
      hisRecoVertexVsGenVertexPt_->Fill(inputData.genPt());
    }

    if (settings_.debug() > 2) {
      edm::LogInfo("VertexAnalyzer") << "analyzer::** RECO VERTICES **";
      for (RecoVertexWithTP* vertex : recoVertices) {
        edm::LogInfo("VertexAnalyzer") << "analyzer::recovertex z0 " << vertex->z0() << " pt " << vertex->pT()
                                       << " highpt " << vertex->hasHighPt() << " numtracks " << vertex->numTracks()
                                       << " numTrueTracks " << vertex->numTrueTracks();
      }
      edm::LogInfo("VertexAnalyzer") << "analyzer::True PrimaryVertex z0 " << TruePrimaryVertex.z0() << " pT "
                                     << TruePrimaryVertex.pT();
      edm::LogInfo("VertexAnalyzer") << "analyzer::Reco PrimaryVertex z0 " << RecoPrimaryVertex->z0() << " pT "
                                     << RecoPrimaryVertex->pT() << " nTracks " << RecoPrimaryVertex->numTracks();
    }

    unsigned int TrackRank = 0;
    for (unsigned int id = 0; id < numVertices; ++id) {
      if (id == primaryVertexIndex)
        continue;
      if (recoVertices.at(id)->numTrueTracks() > RecoPrimaryVertex->numTrueTracks()) {
        TrackRank++;
      }
    }

    hisRecoVertexTrackRank_->Fill(TrackRank);
    hisCorrelatorInputTracks_->Fill(numInputTracks);
    hisCorrelatorTPInputTracks_->Fill(numInputTracks);

    // TODO: REVIEW
    // hisNumVxIterations_->Fill(vf.NumIterations());
    // hisNumVxIterationsPerTrack_->Fill(vf.IterationsPerTrack());

    hisNoRecoVertices_->Fill(numVertices);
    hisNoPileUpVertices_->Fill(inputData.getRecoPileUpVertices().size());
    hisNoRecoVsNoTruePileUpVertices_->Fill(numVertices, inputData.getRecoPileUpVertices().size());

    if (TruePrimaryVertex.numTracks() > 0)
      hisPrimaryVertexTrueZ0_->Fill(TruePrimaryVertex.z0());

    float z0res = TruePrimaryVertex.z0() - RecoPrimaryVertex->z0();
    float pTres = std::abs(TruePrimaryVertex.pT() - RecoPrimaryVertex->pT());
    hisRecoVertexZ0Resolution_->Fill(z0res);
    hisRecoVertexZ0Separation_->Fill(std::abs(z0res));

    // Vertex has been found
    if (std::abs(z0res) < settings_.vx_resolution()) {
      if (settings_.debug() > 2) {
        edm::LogInfo("VertexAnalyzer") << "analyzer::** RECO TRACKS in PV **";
        for (const L1TrackTruthMatched* track : RecoPrimaryVertex->tracks()) {
          if (track->getMatchedTP() != nullptr) {
            edm::LogInfo("VertexAnalyzer")
                << "analyzer::matched TP " << getIndex(inputData.getTPs(), *track->getMatchedTP());
          }
          edm::LogInfo("VertexAnalyzer") << "analyzer::pT " << track->pt() << " phi0 " << track->phi0() << " z0 "
                                         << track->z0();
        }

        edm::LogInfo("VertexAnalyzer") << "analyzer::** TRUE TRACKS in PV **";
        for (const TP& track : TruePrimaryVertex.tracks()) {
          edm::LogInfo("VertexAnalyzer") << "analyzer::index " << getIndex(inputData.getTPs(), track) << " pT "
                                         << track->pt() << " phi0 " << track->phi() << " z0 " << track->z0()
                                         << " status " << track.physicsCollision();
        }
      }

      if (RecoPrimaryVertex->pT() > 100.) {
        hisRecoGenuineVertexVsGenTkVertexPt_->Fill(TruePrimaryVertex.pT());
      }

      hisRecoVertexVsNumGenTracks_->Fill(TruePrimaryVertex.numTracks());
      hisRecoVertexVsGenTkVertexPtForEff_->Fill(TruePrimaryVertex.pT());
      hisRecoPrimaryVertexVsTrueZ0_->Fill(TruePrimaryVertex.z0());

      // ** Reconstructed Primary Vertex Histos **
      hisRecoVertexPTResolution_->Fill(pTres / TruePrimaryVertex.pT());
      hisRecoVertexPTResolutionVsTruePt_->Fill(TruePrimaryVertex.pT(), pTres / TruePrimaryVertex.pT());

      hisRecoVertexPTVsTruePt_->Fill(RecoPrimaryVertex->pT(), TruePrimaryVertex.pT());
      hisNoTracksFromPrimaryVertex_->Fill(RecoPrimaryVertex->numTracks(), TruePrimaryVertex.numTracks());
      hisNoTrueTracksFromPrimaryVertex_->Fill(RecoPrimaryVertex->numTrueTracks(), TruePrimaryVertex.numTracks());
      hisRecoPrimaryVertexZ0width_->Fill(RecoPrimaryVertex->z0width());
      hisRecoVertexPT_->Fill(RecoPrimaryVertex->pT());

      float matchratio = float(RecoPrimaryVertex->numTrueTracks()) / float(TruePrimaryVertex.numTracks());
      if (matchratio > 1.)
        matchratio = 1.;
      hisRatioMatchedTracksInPV_->Fill(matchratio);
      float trueRate = float(RecoPrimaryVertex->numTrueTracks()) / float(RecoPrimaryVertex->numTracks());
      hisTrueTracksRateInPV_->Fill(trueRate);
      float fakeRate = float(RecoPrimaryVertex->numTracks() - RecoPrimaryVertex->numTrueTracks()) /
                       float(RecoPrimaryVertex->numTracks());
      hisFakeTracksRateInPV_->Fill(fakeRate);
      hisRecoPrimaryVertexResolutionVsTrueZ0_->Fill(TruePrimaryVertex.z0(), std::abs(z0res));
    } else {
      hisRecoVertexOffPT_->Fill(RecoPrimaryVertex->pT());
      hisUnmatchedVertexZ0distance_->Fill(std::abs(z0res));
      if (settings_.debug() > 2) {
        edm::LogInfo("VertexAnalyzer")
            << "analyzer::Vertex Reconstruction Algorithm doesn't find the correct primary vertex (Delta Z = "
            << std::abs(z0res) << ")";
      }
    }

    if (settings_.debug() > 2) {
      for (const L1TrackTruthMatched* l1track : RecoPrimaryVertex->tracks()) {
        if (l1track->getMatchedTP() == nullptr) {
          edm::LogInfo("VertexAnalyzer") << "analyzer::FAKE track assigned to PV. Track z0: " << l1track->z0()
                                         << " track pT " << l1track->pt() << " chi2/ndof " << l1track->chi2dof()
                                         << " numstubs " << l1track->getNumStubs();
        } else if (l1track->getMatchedTP()->physicsCollision() == 0) {
          edm::LogInfo("VertexAnalyzer") << "analyzer::Pile-Up track assigned to PV. Track z0: " << l1track->z0()
                                         << " track pT " << l1track->pt();
        } else {
          edm::LogInfo("VertexAnalyzer") << "analyzer::Physics Collision track assigned to PV. Track z0: "
                                         << l1track->z0() << " track pT " << l1track->pt() << " numstubs "
                                         << l1track->getNumStubs() << "\n (real values) id: "
                                         << getIndex(inputData.getTPs(), *l1track->getMatchedTP()) << " pT "
                                         << (*l1track->getMatchedTP())->pt() << " eta "
                                         << (*l1track->getMatchedTP())->eta() << " d0 "
                                         << (*l1track->getMatchedTP())->d0() << " z0 "
                                         << (*l1track->getMatchedTP())->z0() << " physicsCollision "
                                         << l1track->getMatchedTP()->physicsCollision() << " useForEff() "
                                         << l1track->getMatchedTP()->useForEff() << " pdg "
                                         << (*l1track->getMatchedTP())->pdgId();
        }
      }
    }

    unsigned int lostTracks = 0;
    unsigned int misassignedTracks = 0;

    /**
   * TODO: REVIEW
   *
   * The current implementation of the "matching" of unmatched TPs to fitted
   * tracks takes the first L1TrackTruthMatched object that is associated with a
   * given unmatched TP. In the original implementation of below code (when it
   * was coupled to the vertex producer) the fitted tracks were coming
   * directly from the vertex finder which would often sort them in pT. As a
   * result if one is to compare this implementation with the version where
   * below code was part of the histogramming class which was initiated from
   * the producer, minor differences are to be expected (as a different fitted
   * track will sometimes be matched to a given TP).
   */
    if (settings_.debug() > 2)
      edm::LogInfo("VertexAnalyzer") << "analyzer::*** Misassigned primary vertex tracks ***";
    for (const TP& tp : TruePrimaryVertex.tracks()) {
      bool found = false;
      for (const L1TrackTruthMatched* l1track : RecoPrimaryVertex->tracks()) {
        if (l1track->getMatchedTP() != nullptr) {
          if (getIndex(inputData.getTPs(), tp) == getIndex(inputData.getTPs(), *l1track->getMatchedTP())) {
            found = true;
            break;
          }
        }
      }

      if (!found) {
        bool TrackIsReconstructed = false;
        for (const L1Track* l1trackBase : l1TrackPtrs) {
          const L1TrackTruthMatched* l1track = trackAssociationMap[l1trackBase->getTTTrackPtr()];
          if (l1track->getMatchedTP() != nullptr) {
            if (getIndex(inputData.getTPs(), tp) == getIndex(inputData.getTPs(), *l1track->getMatchedTP())) {
              TrackIsReconstructed = true;
              hisUnmatchZ0distance_->Fill(std::abs(l1track->z0() - RecoPrimaryVertex->z0()));
              hisUnmatchPt_->Fill(l1track->pt());
              hisUnmatchEta_->Fill(l1track->eta());
              hisUnmatchTruePt_->Fill(tp->pt());
              hisUnmatchTrueEta_->Fill(tp->eta());

              double mindistance = 999.;
              for (const L1TrackTruthMatched* vertexTrack : RecoPrimaryVertex->tracks()) {
                if (std::abs(vertexTrack->z0() - l1track->z0()) < mindistance)
                  mindistance = std::abs(vertexTrack->z0() - l1track->z0());
              }
              hisUnmatchZ0MinDistance_->Fill(mindistance);

              if (settings_.debug() > 1) {
                edm::LogInfo("VertexAnalyzer")
                    << "analyzer::PV Track assigned to wrong vertex. Track z0: " << l1track->z0()
                    << " PV z0: " << RecoPrimaryVertex->z0() << " tp z0 " << tp->z0() << " track pT " << l1track->pt()
                    << " tp pT " << tp->pt() << " tp d0 " << tp->d0() << " track eta " << l1track->eta();
              }
              break;
            }
          }
        }

        if (!TrackIsReconstructed) {
          lostTracks++;
        } else {
          misassignedTracks++;
        }
      }

      found = false;
    }

    hisLostPVtracks_->Fill(lostTracks);
    hisUnmatchedPVtracks_->Fill(misassignedTracks);

    hisPrimaryVertexZ0width_->Fill(TruePrimaryVertex.z0width());

    float z0distance = 0.;

    for (unsigned int i = 0; i < numVertices; ++i) {
      if (i < numVertices - 1) {
        z0distance = recoVertices.at(i + 1)->z0() - recoVertices.at(i)->z0();
        hisRecoVertexZ0Spacing_->Fill(z0distance);
      }
      if (i != primaryVertexIndex) {
        hisRecoPileUpVertexZ0width_->Fill(recoVertices.at(i)->z0width());
        hisRecoPileUpVertexPT_->Fill(recoVertices.at(i)->pT());
        double PUres = 999.;
        for (unsigned int j = 0; j < inputData.getRecoPileUpVertices().size(); ++j) {
          if (std::abs(recoVertices.at(i)->z0() - inputData.getRecoPileUpVertices()[j].z0()) < PUres) {
            PUres = std::abs(recoVertices.at(i)->z0() - inputData.getRecoPileUpVertices()[j].z0());
          }
        }
        hisRecoPileUpVertexZ0resolution_->Fill(PUres);
      }
    }

    for (unsigned int i = 0; i < inputData.getRecoPileUpVertices().size(); ++i) {
      if (i < inputData.getRecoPileUpVertices().size() - 1) {
        z0distance = inputData.getRecoPileUpVertices()[i + 1].z0() - inputData.getRecoPileUpVertices()[i].z0();
        hisPileUpVertexZ0Spacing_->Fill(z0distance);
      }
      hisPileUpVertexZ0_->Fill(inputData.getRecoPileUpVertices()[i].z0());
      hisPileUpVertexZ0width_->Fill(inputData.getRecoPileUpVertices()[i].z0width());
    }

    if (settings_.debug() > 2)
      edm::LogInfo("VertexAnalyzer") << "analyzer::================ End of Event ==============";

    if (printResults_) {
      edm::LogInfo("VertexAnalyzer") << "analyzer::" << numVertices << " vertices were found ... ";
      for (const auto& vtx : recoVertices) {
        edm::LogInfo("VertexAnalyzer") << "analyzer::  * z0 = " << vtx->z0() << "; contains " << vtx->numTracks()
                                       << " tracks ...";
        for (const auto& trackPtr : vtx->tracks())
          edm::LogInfo("VertexAnalyzer") << "analyzer::     - z0 = " << trackPtr->z0() << "; pt = " << trackPtr->pt()
                                         << ", eta = " << trackPtr->eta() << ", phi = " << trackPtr->phi0();
      }
    }
  }

  void VertexAnalyzer::endJob() {
    // Vertex Efficiency
    edm::Service<TFileService> fs_;
    TFileDirectory inputDir = fs_->mkdir("VertexEfficiency");

    PVefficiencyVsTrueZ0_ = inputDir.make<TEfficiency>(*hisRecoPrimaryVertexVsTrueZ0_, *hisPrimaryVertexTrueZ0_);
    PVefficiencyVsTrueZ0_->SetNameTitle("PVefficiencyVsTrueZ0_",
                                        "Primary Vertex Finding Efficiency; true z_{0}; Efficiency");

    edm::LogInfo("VertexAnalyzer") << "analyzer::==================== VERTEX RECONSTRUCTION ======================\n"
                                   << "Average no. Reconstructed Vertices: " << hisNoRecoVertices_->GetMean() << "("
                                   << hisNoRecoVertices_->GetMean() * 100. / (hisNoPileUpVertices_->GetMean() + 1.)
                                   << "%)\n"
                                   << "Average ratio of matched tracks in primary vertex "
                                   << hisRatioMatchedTracksInPV_->GetMean() * 100 << " %\n"
                                   << "Averate ratio of fake tracks in primary vertex "
                                   << hisFakeTracksRateInPV_->GetMean() * 100 << " %\n"
                                   << "Average PV z0 separation " << hisRecoVertexZ0Separation_->GetMean() << " cm\n"
                                   << "PV z0 resolution " << hisRecoVertexZ0Resolution_->GetStdDev() << " cm ";

    float recoPVeff =
        double(hisRecoPrimaryVertexVsTrueZ0_->GetEntries()) / double(hisPrimaryVertexTrueZ0_->GetEntries());
    float numRecoPV = double(hisRecoPrimaryVertexVsTrueZ0_->GetEntries());
    float numPVs = double(hisPrimaryVertexTrueZ0_->GetEntries());

    float recoPVeff_err = sqrt((numRecoPV + 1) * (numRecoPV + 2) / ((numPVs + 2) * (numPVs + 3)) -
                               (numRecoPV + 1) * (numRecoPV + 1) / ((numPVs + 2) * (numPVs + 2)));

    edm::LogInfo("VertexAnalyzer") << "analyzer::PrimaryVertex Finding Efficiency = " << recoPVeff << " +/- "
                                   << recoPVeff_err;
  };

  VertexAnalyzer::~VertexAnalyzer(){};
}  // namespace l1tVertexFinder

using namespace l1tVertexFinder;

// define this as a plug-in
DEFINE_FWK_MODULE(VertexAnalyzer);
