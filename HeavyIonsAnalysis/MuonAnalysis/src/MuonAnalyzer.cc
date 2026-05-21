#include "HeavyIonsAnalysis/MuonAnalysis/interface/MuonAnalyzer.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

using namespace std::placeholders;
using namespace std;
using namespace reco;
using namespace edm;
using namespace HepMC;

MuonAnalyzer::MuonAnalyzer(const edm::ParameterSet& ps) {
  doGen_ = ps.getParameter<bool>("doGen");
  doReco_ = ps.getUntrackedParameter<bool>("doReco");

  vertexToken_ = consumes<std::vector<reco::Vertex>>(ps.getParameter<edm::InputTag>("vertexSrc"));
  genToken_ = consumes<edm::View<pat::PackedGenParticle>>(ps.getParameter<edm::InputTag>("genparticle"));
  muonToken_ = consumes<edm::View<pat::Muon>>(ps.getParameter<edm::InputTag>("muonSrc"));
  trackBuilderToken_ = esConsumes(edm::ESInputTag("", "TransientTrackBuilder"));

  // initialize output TTree
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("MuonTree", "muon tree");

  tree_->Branch("run", &run_);
  tree_->Branch("event", &event_);
  tree_->Branch("lumi", &lumi_);

  tree_->Branch("nGen", &nGen_);
  tree_->Branch("genVtx_x", &genVtx_x_);
  tree_->Branch("genVtx_y", &genVtx_y_);
  tree_->Branch("genVtx_z", &genVtx_z_);
  tree_->Branch("genPID", &genPID_);
  tree_->Branch("genStatus", &genStatus_);
  tree_->Branch("genP", &genP_);
  tree_->Branch("genPt", &genPt_);
  tree_->Branch("genEta", &genEta_);
  tree_->Branch("genPhi", &genPhi_);
  tree_->Branch("genMotherID", &genMotherID_);

  // reco muon candidates
  tree_->Branch("nReco", &nReco_);
  tree_->Branch("recoP", &recoP_);
  tree_->Branch("recoPt", &recoPt_);
  tree_->Branch("recoEta", &recoEta_);
  tree_->Branch("recoPhi", &recoPhi_);
  tree_->Branch("recoL1Eta", &recoL1Eta_);
  tree_->Branch("recoL1Phi", &recoL1Phi_);
  tree_->Branch("recoCharge", &recoCharge_);
  tree_->Branch("recoType", &recoType_);
  tree_->Branch("recoIsGood", &recoIsGood_);
  tree_->Branch("recoIsGlobal", &recoIsGlobal_);
  tree_->Branch("recoIsTracker", &recoIsTracker_);
  tree_->Branch("recoIsPF", &recoIsPF_);
  tree_->Branch("recoIsSTA", &recoIsSTA_);
  tree_->Branch("recoDxy", &recoDxy_);
  tree_->Branch("recoDz", &recoDz_);
  tree_->Branch("recoDxyErr", &recoDxyErr_);
  tree_->Branch("recoDzErr", &recoDzErr_);
  tree_->Branch("recoIP3D", &recoIP3D_);
  tree_->Branch("recoIP3DErr", &recoIP3DErr_);
  tree_->Branch("recoNMatchedStations", &recoNMatchedStations_);
  tree_->Branch("recoIsoTrk", &recoIsoTrk_);
  tree_->Branch("recoPFChIso", &recoPFChIso_);
  tree_->Branch("recoPFPhoIso", &recoPFPhoIso_);
  tree_->Branch("recoPFNeuIso", &recoPFNeuIso_);
  tree_->Branch("recoPFPUIso", &recoPFPUIso_);
  tree_->Branch("recoMVAIso", &recoMVAIso_);
  for (auto& w : recoMVAIsoWP_)
    tree_->Branch(("recoMVAIso"+w.first).c_str(), &(w.second));
  tree_->Branch("recoIDHybridSoft", &recoIDHybridSoft_);
  tree_->Branch("recoIDSoft", &recoIDSoft_);
  tree_->Branch("recoIDLoose", &recoIDLoose_);
  tree_->Branch("recoIDMedium", &recoIDMedium_);
  tree_->Branch("recoIDMediumPrompt", &recoIDMediumPrompt_);
  tree_->Branch("recoIDTight", &recoIDTight_);
  tree_->Branch("recoIDGlobalHighPt", &recoIDGlobalHighPt_);
  tree_->Branch("recoIDTrkHighPt", &recoIDTrkHighPt_);
  tree_->Branch("recoIDInTime", &recoIDInTime_);
  tree_->Branch("recoMVAIDSoft", &recoMVAIDSoft_);
  tree_->Branch("recoMVAIDLoose", &recoMVAIDLoose_);
  tree_->Branch("recoMVAIDMedium", &recoMVAIDMedium_);
  tree_->Branch("recoMVAIDTight", &recoMVAIDTight_);
  tree_->Branch("recoMVAIDLooseLowPt", &recoMVAIDLooseLowPt_);
  tree_->Branch("recoMVAIDMediumLowPt", &recoMVAIDMediumLowPt_);

  // inner tracks
  tree_->Branch("nInner", &nInner_);
  tree_->Branch("innerDxy", &innerDxy_);
  tree_->Branch("innerDz", &innerDz_);
  tree_->Branch("innerDxyErr", &innerDxyErr_);
  tree_->Branch("innerDzErr", &innerDzErr_);
  tree_->Branch("innerP", &innerP_);
  tree_->Branch("innerPt", &innerPt_);
  tree_->Branch("innerPtErr", &innerPtErr_);
  tree_->Branch("innerEta", &innerEta_);
  tree_->Branch("innerTrkLayers", &innerTrkLayers_);
  tree_->Branch("innerNTrkHits", &innerNTrkHits_);
  tree_->Branch("innerPixelLayers", &innerPixelLayers_);
  tree_->Branch("innerNPixelHits", &innerNPixelHits_);
  tree_->Branch("innerIsHighPurityTrack", &innerIsHighPurityTrack_);
  tree_->Branch("innerNormChi2", &innerNormChi2_);

  // global muons
  tree_->Branch("nGlobal", &nGlobal_);
  tree_->Branch("globalP", &globalP_);
  tree_->Branch("globalPt", &globalPt_);
  tree_->Branch("globalPtErr", &globalPtErr_);
  tree_->Branch("globalEta", &globalEta_);
  tree_->Branch("globalIsArbitrated", &globalIsArbitrated_);
  tree_->Branch("globalDxy", &globalDxy_);
  tree_->Branch("globalDz", &globalDz_);
  tree_->Branch("globalDxyErr", &globalDxyErr_);
  tree_->Branch("globalDzErr", &globalDzErr_);
  tree_->Branch("globalNormChi2", &globalNormChi2_);
  tree_->Branch("globalNMuonHits", &globalNMuonHits_);
}

MuonAnalyzer::~MuonAnalyzer() {}

void MuonAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& es) {
  // cleanup from previous event

  if (doGen_) {
    nGen_ = 0;
    genVtx_x_.clear();
    genVtx_y_.clear();
    genVtx_z_.clear();
    genPID_.clear();
    genStatus_.clear();
    genP_.clear();
    genPt_.clear();
    genEta_.clear();
    genPhi_.clear();
    genMotherID_.clear();
  }

  nReco_ = 0;
  recoP_.clear();
  recoPt_.clear();
  recoEta_.clear();
  recoPhi_.clear();
  recoL1Eta_.clear();
  recoL1Phi_.clear();
  recoCharge_.clear();
  recoType_.clear();
  recoIsGood_.clear();
  recoIsGlobal_.clear();
  recoIsTracker_.clear();
  recoIsPF_.clear();
  recoIsSTA_.clear();
  recoDxy_.clear();
  recoDz_.clear();
  recoDxyErr_.clear();
  recoDzErr_.clear();
  recoIP3D_.clear();
  recoIP3DErr_.clear();
  recoNMatchedStations_.clear();
  recoIsoTrk_.clear();
  recoPFChIso_.clear();
  recoPFPhoIso_.clear();
  recoPFNeuIso_.clear();
  recoPFPUIso_.clear();
  recoMVAIso_.clear();
  for (auto& w : recoMVAIsoWP_)
    w.second.clear();
  recoIDHybridSoft_.clear();
  recoIDSoft_.clear();
  recoIDLoose_.clear();
  recoIDMedium_.clear();
  recoIDMediumPrompt_.clear();
  recoIDTight_.clear();
  recoIDGlobalHighPt_.clear();
  recoIDTrkHighPt_.clear();
  recoIDInTime_.clear();

  recoMVAIDSoft_.clear();
  recoMVAIDLoose_.clear();
  recoMVAIDMedium_.clear();
  recoMVAIDTight_.clear();
  recoMVAIDLooseLowPt_.clear();
  recoMVAIDMediumLowPt_.clear();

  nInner_ = 0;
  innerDxy_.clear();
  innerDz_.clear();
  innerDxyErr_.clear();
  innerDzErr_.clear();
  innerP_.clear();
  innerPt_.clear();
  innerPtErr_.clear();
  innerEta_.clear();
  innerTrkLayers_.clear();
  innerNTrkHits_.clear();
  innerPixelLayers_.clear();
  innerNPixelHits_.clear();
  innerIsHighPurityTrack_.clear();
  innerNormChi2_.clear();

  nGlobal_ = 0;
  globalP_.clear();
  globalPt_.clear();
  globalPtErr_.clear();
  globalEta_.clear();
  globalIsArbitrated_.clear();
  globalDxy_.clear();
  globalDz_.clear();
  globalDxyErr_.clear();
  globalDzErr_.clear();
  globalNormChi2_.clear();
  globalNMuonHits_.clear();

  run_ = e.id().run();
  event_ = e.id().event();
  lumi_ = e.luminosityBlock();

  if (doGen_) {
    edm::Handle<edm::View<pat::PackedGenParticle>> genCollection;
    e.getByToken(genToken_, genCollection);

    if (genCollection.isValid()) {
      for (unsigned i = 0; i < genCollection->size(); ++i) {
        pat::PackedGenParticle const& genParticle = genCollection->at(i);
        if (!(abs(genParticle.pdgId()) == 13 && genParticle.status() == 1))
          continue;  // only muons

        nGen_++;

        genVtx_x_.push_back(genParticle.vx());
        genVtx_y_.push_back(genParticle.vy());
        genVtx_z_.push_back(genParticle.vz());

        genPID_.push_back(genParticle.pdgId());
        genStatus_.push_back(genParticle.status());
        genP_.push_back(genParticle.p());
        genPt_.push_back(genParticle.pt());
        genEta_.push_back(genParticle.eta());
        genPhi_.push_back(genParticle.phi());

        // search for the mother ID (if any)
        int motherID = genParticle.pdgId();  // initialize with the muon ID in case of single particle gun

        if (genParticle.numberOfMothers() > 0) {
          bool isQuarkoniumMother = false;

          for (unsigned int mom = 0; mom < genParticle.numberOfMothers(); mom++) {
            int momID = genParticle.mother(mom)->pdgId();
            if (momID == 443 || momID == 100443 || momID == 553 || momID == 100553 || momID == 200553) {
              isQuarkoniumMother = true;
              motherID = momID;
              break;  // mother found!
            }
          }

          // if the mother is not a quarkonium state, check the grandmother
          if (!isQuarkoniumMother) {
            const Candidate* motherCand = genParticle.mother(0);

            for (unsigned int gmom = 0; gmom < motherCand->numberOfMothers(); gmom++) {
              int gmomID = motherCand->mother(gmom)->pdgId();
              if (gmomID == 443 || gmomID == 100443 || gmomID == 553 || gmomID == 100553 || gmomID == 200553) {
                motherID = gmomID;
                break;  // mother found!
              }
            }
          }
        }

        genMotherID_.push_back(motherID);
      }  // end of gen particle loop
    }
  }  // end of doGen_

  if (doReco_) {
    edm::Handle<std::vector<reco::Vertex>> vertices;
    e.getByToken(vertexToken_, vertices);

    // best-known primary vertex coordinates
    reco::Vertex pv(math::XYZPoint(0, 0, -999), math::Error<3>::type());
    for (const auto& vertex : *vertices) {
      if (!vertex.isFake()) {
        pv = vertex;
        break;
      }
    }

    const auto& tb = es.getHandle(trackBuilderToken_);

    // fill tree branches with reconstructed muons.
    edm::Handle<edm::View<pat::Muon>> recoMuons;
    e.getByToken(muonToken_, recoMuons);

    for (const auto& mu : *recoMuons) {
      if (!(mu.isPFMuon() || mu.isGlobalMuon() || mu.isTrackerMuon()))
        continue;

      nReco_++;

      recoP_.push_back(mu.p());
      recoPt_.push_back(mu.pt());
      recoEta_.push_back(mu.eta());
      recoPhi_.push_back(mu.phi());
      recoL1Eta_.push_back(mu.hasUserFloat("l1Eta") ? mu.userFloat("l1Eta") : -99);
      recoL1Phi_.push_back(mu.hasUserFloat("l1Phi") ? mu.userFloat("l1Phi") : -99);
      recoCharge_.push_back(mu.charge());
      recoType_.push_back(mu.type());
      recoIsGood_.push_back(muon::isGoodMuon(mu, muon::selectionTypeFromString("TMOneStationTight")));

      recoIsGlobal_.push_back(mu.isGlobalMuon());
      recoIsTracker_.push_back(mu.isTrackerMuon());
      recoIsPF_.push_back(mu.isPFMuon());
      recoIsSTA_.push_back(mu.isStandAloneMuon());

      recoDxy_.push_back(mu.muonBestTrack()->dxy(pv.position()));
      recoDz_.push_back(mu.muonBestTrack()->dz(pv.position()));
      recoDxyErr_.push_back(mu.muonBestTrack()->dxyError());
      recoDzErr_.push_back(mu.muonBestTrack()->dzError());

      // initialize with unphysical values
      float recoIP3D = -999;
      float recoIP3DErr = -999;
      if (pv.isValid()) {
        // 3D impact parameter
        reco::TransientTrack tt = tb->build(mu.muonBestTrack().get());
        recoIP3D = IPTools::absoluteImpactParameter3D(tt, pv).second.value();
        recoIP3DErr = IPTools::absoluteImpactParameter3D(tt, pv).second.error();
      }
      recoIP3D_.push_back(recoIP3D);
      recoIP3DErr_.push_back(recoIP3DErr);

      // inner track info
      if (mu.innerTrack().isNonnull()) {
        const reco::TrackRef innMu = mu.innerTrack();

        const reco::HitPattern& hitPat = innMu->hitPattern();

        nInner_++;

        innerDxy_.push_back(innMu->dxy(pv.position()));
        innerDz_.push_back(innMu->dz(pv.position()));
        innerDxyErr_.push_back(innMu->dxyError());
        innerDzErr_.push_back(innMu->dzError());

        innerP_.push_back(innMu->p());
        innerPt_.push_back(innMu->pt());
        innerPtErr_.push_back(innMu->ptError());
        innerEta_.push_back(innMu->eta());

        innerTrkLayers_.push_back(hitPat.trackerLayersWithMeasurement());
        innerNTrkHits_.push_back(hitPat.numberOfValidTrackerHits());

        innerPixelLayers_.push_back(hitPat.pixelLayersWithMeasurement());
        innerNPixelHits_.push_back(hitPat.numberOfValidPixelHits());

        innerIsHighPurityTrack_.push_back(innMu->quality(reco::TrackBase::highPurity));

        innerNormChi2_.push_back(innMu->normalizedChi2());

      } else {
        innerDxy_.push_back(-99);
        innerDz_.push_back(-99);
        innerDxyErr_.push_back(-99);
        innerDzErr_.push_back(-99);

        innerP_.push_back(-99);
        innerPt_.push_back(-99);
        innerPtErr_.push_back(-99);
        innerEta_.push_back(-99);

        innerTrkLayers_.push_back(-99);
        innerNTrkHits_.push_back(-99);
        innerPixelLayers_.push_back(-99);
        innerNPixelHits_.push_back(-99);

        innerIsHighPurityTrack_.push_back(false);

        innerNormChi2_.push_back(-99);
      }

      // global muons, avoiding overlaps with inner track variables if possible
      if (mu.globalTrack().isNonnull() && mu.isGlobalMuon()) {
          const reco::TrackRef glbMu = mu.globalTrack();

          nGlobal_++;

          globalP_.push_back(glbMu->p());
          globalPt_.push_back(glbMu->pt());
          globalPtErr_.push_back(glbMu->ptError());
          globalEta_.push_back(glbMu->eta());

          globalIsArbitrated_.push_back(muon::isGoodMuon(mu, muon::selectionTypeFromString("TrackerMuonArbitrated")));

          globalDxy_.push_back(glbMu->dxy(pv.position()));
          globalDz_.push_back(glbMu->dz(pv.position()));
          globalDxyErr_.push_back(glbMu->dxyError());
          globalDzErr_.push_back(glbMu->dzError());

          globalNormChi2_.push_back(glbMu->normalizedChi2());
          globalNMuonHits_.push_back(glbMu->hitPattern().numberOfValidMuonHits());

        } else {
          globalP_.push_back(-99);
          globalPt_.push_back(-99);
          globalPtErr_.push_back(-99);
          globalEta_.push_back(-99);

          globalIsArbitrated_.push_back(false);

          globalDxy_.push_back(-99);
          globalDz_.push_back(-99);
          globalDxyErr_.push_back(-99);
          globalDzErr_.push_back(-99);

          globalNormChi2_.push_back(-99);
          globalNMuonHits_.push_back(-99);
        }

      recoNMatchedStations_.push_back(mu.numberOfMatchedStations());
      recoIsoTrk_.push_back(mu.isolationR03().sumPt);
      recoPFChIso_.push_back(mu.pfIsolationR04().sumChargedHadronPt);
      recoPFPhoIso_.push_back(mu.pfIsolationR04().sumPhotonEt);
      recoPFNeuIso_.push_back(mu.pfIsolationR04().sumNeutralHadronEt);
      recoPFPUIso_.push_back(mu.pfIsolationR04().sumPUPt);

      recoMVAIso_.push_back(mu.hasUserFloat("hiMVAIso") ? mu.userFloat("hiMVAIso") : -99);
      for (auto& w : recoMVAIsoWP_)
        w.second.push_back(mu.hasUserInt("hiMVAIso"+w.first) && mu.userInt("hiMVAIso"+w.first)>0);

      recoIDHybridSoft_.push_back(mu.isGlobalMuon() && mu.isTrackerMuon() && mu.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 && mu.innerTrack()->hitPattern().pixelLayersWithMeasurement() > 0 && fabs(mu.innerTrack()->dxy(pv.position()) < 0.3) && fabs(mu.innerTrack()->dz(pv.position()) < 20.));

      //  muon selectors available at https://github.com/cms-sw/cmssw/blob/4c9240b33ace61c92c6803f0c4eace9ba06e6c8d/DataFormats/MuonReco/interface/Muon.h#L202
      // Cut-based Ids
      recoIDSoft_.push_back(mu.passed(reco::Muon::SoftCutBasedId));
      recoIDLoose_.push_back(mu.passed(reco::Muon::CutBasedIdLoose));
      recoIDMedium_.push_back(mu.passed(reco::Muon::CutBasedIdMedium));
      recoIDMediumPrompt_.push_back(mu.passed(reco::Muon::CutBasedIdMediumPrompt));
      recoIDTight_.push_back(mu.passed(reco::Muon::CutBasedIdTight));
      recoIDGlobalHighPt_.push_back(mu.passed(reco::Muon::CutBasedIdGlobalHighPt));
      recoIDTrkHighPt_.push_back(mu.passed(reco::Muon::CutBasedIdTrkHighPt));
      recoIDInTime_.push_back(mu.passed(reco::Muon::InTimeMuon));

      // MVA-based Ids
      recoMVAIDSoft_.push_back(mu.passed(reco::Muon::SoftMvaId));
      recoMVAIDLoose_.push_back(mu.passed(reco::Muon::MvaLoose));
      recoMVAIDMedium_.push_back(mu.passed(reco::Muon::MvaMedium));
      recoMVAIDTight_.push_back(mu.passed(reco::Muon::MvaTight));
      recoMVAIDLooseLowPt_.push_back(mu.passed(reco::Muon::LowPtMvaLoose));
      recoMVAIDMediumLowPt_.push_back(mu.passed(reco::Muon::LowPtMvaMedium));

    }  // muons loop
  }    // end of doReco_

  tree_->Fill();
}

DEFINE_FWK_MODULE(MuonAnalyzer);
