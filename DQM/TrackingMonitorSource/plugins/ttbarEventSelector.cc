#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "DQM/TrackingMonitorSource/interface/ttbarEventSelector.h"

using namespace std;
using namespace edm;

ttbarEventSelector::ttbarEventSelector(const edm::ParameterSet& ps)
    : electronTag_(ps.getUntrackedParameter<edm::InputTag>("electronInputTag", edm::InputTag("gedGsfElectrons"))),
      jetsTag_(ps.getUntrackedParameter<edm::InputTag>("jetsInputTag", edm::InputTag("ak4PFJetsCHS"))),
      bjetsTag_(ps.getUntrackedParameter<edm::InputTag>("bjetsInputTag", edm::InputTag("pfDeepCSVJetTags", "probb"))),
      pfmetTag_(ps.getUntrackedParameter<edm::InputTag>("pfmetTag", edm::InputTag("pfMet"))),
      muonTag_(ps.getUntrackedParameter<edm::InputTag>("muonInputTag", edm::InputTag("muons"))),
      bsTag_(ps.getUntrackedParameter<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"))),
      electronToken_(consumes<reco::GsfElectronCollection>(electronTag_)),
      jetsToken_(consumes<reco::PFJetCollection>(jetsTag_)),
      bjetsToken_(consumes<reco::JetTagCollection>(bjetsTag_)),
      pfmetToken_(consumes<reco::PFMETCollection>(pfmetTag_)),
      muonToken_(consumes<reco::MuonCollection>(muonTag_)),
      bsToken_(consumes<reco::BeamSpot>(bsTag_)),
      maxEtaEle_(ps.getUntrackedParameter<double>("maxEta", 2.4)),
      maxEtaMu_(ps.getUntrackedParameter<double>("maxEta", 2.4)),
      minPt_(ps.getUntrackedParameter<double>("minPt", 5)),

      // for Electron only
      maxDeltaPhiInEB_(ps.getUntrackedParameter<double>("maxDeltaPhiInEB", .15)),
      maxDeltaEtaInEB_(ps.getUntrackedParameter<double>("maxDeltaEtaInEB", .007)),
      maxHOEEB_(ps.getUntrackedParameter<double>("maxHOEEB", .12)),
      maxSigmaiEiEEB_(ps.getUntrackedParameter<double>("maxSigmaiEiEEB", .01)),
      maxDeltaPhiInEE_(ps.getUntrackedParameter<double>("maxDeltaPhiInEE", .1)),
      maxDeltaEtaInEE_(ps.getUntrackedParameter<double>("maxDeltaEtaInEE", .009)),
      maxHOEEE_(ps.getUntrackedParameter<double>("maxHOEEB_", .10)),
      maxSigmaiEiEEE_(ps.getUntrackedParameter<double>("maxSigmaiEiEEE", .03)),

      // for Muon only
      minChambers_(ps.getUntrackedParameter<uint32_t>("minChambers", 2)),
      minMatches_(ps.getUntrackedParameter<uint32_t>("minMatches", 2)),
      minMatchedStations_(ps.getUntrackedParameter<double>("minMatchedStations", 2)),

      // for Jets only
      maxEtaHighest_Jets_(ps.getUntrackedParameter<double>("maxEtaHighest_Jets", 2.4)),
      maxEta_Jets_(ps.getUntrackedParameter<double>("maxEta_Jets", 3.0)),

      // for b-tag only
      btagFactor_(ps.getUntrackedParameter<double>("btagFactor", 0.6)),

      maxNormChi2_(ps.getUntrackedParameter<double>("maxNormChi2", 10)),
      maxD0_(ps.getUntrackedParameter<double>("maxD0", 0.02)),
      maxDz_(ps.getUntrackedParameter<double>("maxDz", 20.)),
      minPixelHits_(ps.getUntrackedParameter<uint32_t>("minPixelHits", 1)),
      minStripHits_(ps.getUntrackedParameter<uint32_t>("minStripHits", 8)),
      maxIsoEle_(ps.getUntrackedParameter<double>("maxIso", 0.5)),
      maxIsoMu_(ps.getUntrackedParameter<double>("maxIso", 0.3)),
      minPtHighestMu_(ps.getUntrackedParameter<double>("minPtHighestMu", 24)),
      minPtHighestEle_(ps.getUntrackedParameter<double>("minPtHighestEle", 32)),
      minPtHighest_Jets_(ps.getUntrackedParameter<double>("minPtHighest_Jets", 30)),
      minPt_Jets_(ps.getUntrackedParameter<double>("minPt_Jets", 20)),
      minInvMass_(ps.getUntrackedParameter<double>("minInvMass", 140)),
      maxInvMass_(ps.getUntrackedParameter<double>("maxInvMass", 200)),
      minMet_(ps.getUntrackedParameter<double>("minMet", 50)),
      maxMet_(ps.getUntrackedParameter<double>("maxMet", 80)),
      minWmass_(ps.getUntrackedParameter<double>("minWmass", 50)),
      maxWmass_(ps.getUntrackedParameter<double>("maxWmass", 130)) {}

bool ttbarEventSelector::filter(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // beamspot
  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(bsToken_, beamSpot);

  // Read Electron Collection
  edm::Handle<reco::GsfElectronCollection> electronColl;
  iEvent.getByToken(electronToken_, electronColl);
  int le = 0, lj = 0, lm = 0, lbj = 0;
  std::vector<TLorentzVector> list_ele;
  std::vector<int> chrgeList_ele;

  if (electronColl.isValid()) {
    for (auto const& ele : *electronColl) {
      if (!ele.ecalDriven())
        continue;
      if (ele.pt() < minPt_)
        continue;
      // set a max Eta cut
      if (!(ele.isEB() || ele.isEE()))
        continue;

      double hOverE = ele.hadronicOverEm();
      double sigmaee = ele.sigmaIetaIeta();
      double deltaPhiIn = ele.deltaPhiSuperClusterTrackAtVtx();
      double deltaEtaIn = ele.deltaEtaSuperClusterTrackAtVtx();

      // separate cut for barrel and endcap
      if (ele.isEB()) {
        if (fabs(deltaPhiIn) >= maxDeltaPhiInEB_ && fabs(deltaEtaIn) >= maxDeltaEtaInEB_ && hOverE >= maxHOEEB_ &&
            sigmaee >= maxSigmaiEiEEB_)
          continue;
      } else if (ele.isEE()) {
        if (fabs(deltaPhiIn) >= maxDeltaPhiInEE_ && fabs(deltaEtaIn) >= maxDeltaEtaInEE_ && hOverE >= maxHOEEE_ &&
            sigmaee >= maxSigmaiEiEEE_)
          continue;
      }

      reco::GsfTrackRef trk = ele.gsfTrack();
      if (!trk.isNonnull())
        continue;  // only electrons with tracks
      double chi2 = trk->chi2();
      double ndof = trk->ndof();
      double chbyndof = (ndof > 0) ? chi2 / ndof : 0;
      if (chbyndof >= maxNormChi2_)
        continue;

      double trkd0 = trk->d0();
      if (beamSpot.isValid()) {
        trkd0 = -(trk->dxy(beamSpot->position()));
      } else {
        edm::LogError("ElectronSelector") << "Error >> Failed to get BeamSpot for label: " << bsTag_;
      }
      if (std::fabs(trkd0) >= maxD0_)
        continue;

      const reco::HitPattern& hitp = trk->hitPattern();
      int nPixelHits = hitp.numberOfValidPixelHits();
      if (nPixelHits < minPixelHits_)
        continue;

      int nStripHits = hitp.numberOfValidStripHits();
      if (nStripHits < minStripHits_)
        continue;

      // PF Isolation
      reco::GsfElectron::PflowIsolationVariables pfIso = ele.pfIsolationVariables();
      float absiso =
          pfIso.sumChargedHadronPt + std::max(0.0, pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt - 0.5 * pfIso.sumPUPt);
      float eiso = absiso / (ele.pt());
      if (eiso > maxIsoEle_)
        continue;

      TLorentzVector lv_ele;
      lv_ele.SetPtEtaPhiE(ele.pt(), ele.eta(), ele.phi(), ele.energy());
      list_ele.push_back(lv_ele);
      le = list_ele.size();
      chrgeList_ele.push_back(ele.charge());
    }
    le = list_ele.size();
  } else {
    edm::LogError("ElectronSelector") << "Error >> Failed to get ElectronCollection for label: " << electronTag_;
  }

  // Read Muon Collection
  edm::Handle<reco::MuonCollection> muonColl;
  iEvent.getByToken(muonToken_, muonColl);

  std::vector<TLorentzVector> list_mu;
  std::vector<int> chrgeList_mu;
  if (muonColl.isValid()) {
    for (auto const& mu : *muonColl) {
      if (!mu.isGlobalMuon())
        continue;
      if (!mu.isPFMuon())
        continue;
      if (std::fabs(mu.eta()) >= maxEtaMu_)
        continue;
      if (mu.pt() < minPt_)
        continue;

      reco::TrackRef gtk = mu.globalTrack();
      double chi2 = gtk->chi2();
      double ndof = gtk->ndof();
      double chbyndof = (ndof > 0) ? chi2 / ndof : 0;
      if (chbyndof >= maxNormChi2_)
        continue;

      reco::TrackRef tk = mu.innerTrack();
      if (beamSpot.isValid()) {
        double trkd0 = -(tk->dxy(beamSpot->position()));
        if (std::fabs(trkd0) >= maxD0_)
          continue;
        double trkdz = tk->dz(beamSpot->position());
        if (std::fabs(trkdz) >= maxDz_)
          continue;
      } else {
        edm::LogError("MuonSelector") << "Error >> Failed to get BeamSpot for label: " << bsTag_;
      }

      const reco::HitPattern& hitp = gtk->hitPattern();
      if (hitp.numberOfValidPixelHits() < minPixelHits_)
        continue;
      if (hitp.numberOfValidStripHits() < minStripHits_)
        continue;

      // Hits/section in the muon chamber
      if (mu.numberOfChambers() < minChambers_)
        continue;
      if (mu.numberOfMatches() < minMatches_)
        continue;
      if (mu.numberOfMatchedStations() < minMatchedStations_)
        continue;
      if (!muon::isGoodMuon(mu, muon::GlobalMuonPromptTight))
        continue;

      // PF Isolation
      const reco::MuonPFIsolation& pfIso04 = mu.pfIsolationR04();
      double absiso = pfIso04.sumChargedHadronPt +
                      std::max(0.0, pfIso04.sumNeutralHadronEt + pfIso04.sumPhotonEt - 0.5 * pfIso04.sumPUPt);
      if (absiso / mu.pt() > maxIsoMu_)
        continue;

      TLorentzVector lv_mu;
      lv_mu.SetPtEtaPhiE(mu.pt(), mu.eta(), mu.phi(), mu.energy());
      list_mu.push_back(lv_mu);
      lm = list_mu.size();
      chrgeList_mu.push_back(mu.charge());
    }
    lm = list_mu.size();
  } else {
    edm::LogError("MuonSelector") << "Error >> Failed to get MuonCollection for label: " << muonTag_;
    return false;
  }

  // for Track jet collections
  edm::Handle<reco::PFJetCollection> jetColl;
  iEvent.getByToken(jetsToken_, jetColl);
  std::vector<TLorentzVector> list_jets;

  if (jetColl.isValid()) {
    for (const auto& jets : *jetColl) {
      if (jets.pt() < minPt_Jets_)
        continue;
      if (std::fabs(jets.eta()) > maxEta_Jets_)
        continue;
      TLorentzVector lv_jets;  // lv_bJets;
      lv_jets.SetPtEtaPhiE(jets.pt(), jets.eta(), jets.phi(), jets.energy());
      list_jets.push_back(lv_jets);
    }
    lj = list_jets.size();
  }

  edm::Handle<reco::JetTagCollection> bTagHandle;
  iEvent.getByToken(bjetsToken_, bTagHandle);
  const reco::JetTagCollection& bTags = *(bTagHandle.product());
  std::vector<TLorentzVector> list_bjets;
  std::cout << "nbjets " << bTags.size() << std::endl;

  if (!bTags.empty()) {
    for (unsigned bj = 0; bj != bTags.size(); ++bj) {
      TLorentzVector lv_bjets;
      lv_bjets.SetPtEtaPhiE(
          bTags[bj].first->pt(), bTags[bj].first->eta(), bTags[bj].first->phi(), bTags[bj].first->energy());
      if ((bTags[bj].second > btagFactor_) && (lv_bjets.Pt() > minPt_Jets_))
        list_bjets.push_back(lv_bjets);
    }
    lbj = list_bjets.size();
  }

  std::cout << "le " << le << "\tlm " << lm << "\tlj " << lj << "\tlbj " << lbj << std::endl;

  // for MET collection
  edm::Handle<reco::PFMETCollection> pfColl;
  iEvent.getByToken(pfmetToken_, pfColl);
  if (EventCategory(le, lm, lj, lbj) == 11) {  // dilepton- ele ele
    if (list_ele[0].Pt() < minPtHighestEle_)
      return false;
    if ((list_ele[0].Pt() < list_mu[0].Pt()) || (list_ele[1].Pt() < list_mu[0].Pt()))
      return false;
    if (chrgeList_ele[0] + chrgeList_ele[1] != 0)
      return false;
    if (pfColl.isValid()) {
      double mt1 = getMt(list_ele[0], pfColl->front());
      double mt2 = getMt(list_ele[1], pfColl->front());
      double mt = mt1 + mt2;
      if (mt < 2 * minMet_ || mt > 2 * maxMet_)
        return false;
    } else {
      edm::LogError("ttbarEventSelector")
          << "Error >> Failed to get PFMETCollection in dilepton ele-ele channel for label: " << pfmetTag_;
      return false;
    }
  } else if (EventCategory(le, lm, lj, lbj) == 12) {  // dilepton - mu mu
    if (list_mu[0].Pt() < minPtHighestMu_)
      return false;
    if ((list_mu[0].Pt() < list_ele[0].Pt()) || (list_mu[1].Pt() < list_ele[0].Pt()))
      return false;
    if (chrgeList_mu[0] + chrgeList_mu[1] != 0)
      return false;
    if (pfColl.isValid()) {
      double mt1 = getMt(list_mu[0], pfColl->front());
      double mt2 = getMt(list_mu[1], pfColl->front());
      double mt = mt1 + mt2;
      if (mt < 2 * minMet_ || mt > 2 * maxMet_)
        return false;
    } else {
      edm::LogError("ttbarEventSelector")
          << "Error >> Failed to get PFMETCollection in dilepton mu-mu channel for label: " << pfmetTag_;
      return false;
    }
  } else if (EventCategory(le, lm, lj, lbj) == 13) {  // dilepton - ele mu
    if ((list_mu[0].Pt() < list_ele[1].Pt()) || (list_ele[0].Pt() < list_mu[1].Pt()))
      return false;
    if ((list_mu[0].Pt() < minPtHighestMu_) || (list_ele[0].Pt() < minPtHighestEle_))
      return false;
    if (chrgeList_mu[0] + chrgeList_ele[0] != 0)
      return false;
    if (pfColl.isValid()) {
      double mt1 = getMt(list_mu[0], pfColl->front());
      double mt2 = getMt(list_ele[0], pfColl->front());
      double mt = mt1 + mt2;
      if (mt < 2 * minMet_ || mt > 2 * maxMet_)
        return false;
    } else {
      edm::LogError("ttbarEventSelector")
          << "Error >> Failed to get PFMETCollection in dilepton ele-mu channel for label: " << pfmetTag_;
      return false;
    }
  }
  if (EventCategory(le, lm, lj, lbj) == 21) {  // semilepton - ele or mu
    if (list_jets[0].Pt() < minPtHighest_Jets_)
      return false;

    if (list_ele[0].Pt() < minPtHighestEle_)
      return false;
    // Both should not be present at the same time
    if ((!list_ele.empty() && list_ele[0].Pt() > minPtHighestEle_) &&
        (!list_mu.empty() && list_mu[0].Pt() > minPtHighestMu_))
      return false;

    return true;
  }
  if (EventCategory(le, lm, lj, lbj) == 22) {  // semilepton - ele or mu
    if (list_jets[0].Pt() < minPtHighest_Jets_)
      return false;

    if (list_mu[0].Pt() < minPtHighestMu_)
      return false;
    // Both should not be present at the same time
    if ((!list_ele.empty() && list_ele[0].Pt() > minPtHighestEle_) &&
        (!list_mu.empty() && list_mu[0].Pt() > minPtHighestMu_))
      return false;

    return true;
  } else if (EventCategory(le, lm, lj, lbj) == 30) {
    if (list_jets[0].Pt() < minPtHighest_Jets_)
      return false;
    for (int i = 0; i < 4; i++) {
      TLorentzVector vjet1;
      for (int j = i + 1; j < 4; j++) {
        TLorentzVector vjet2;
        vjet1 = list_jets[i];
        vjet2 = list_jets[j];
        TLorentzVector vjet = vjet1 + vjet2;
        if (vjet.M() < minWmass_ || vjet.M() > maxWmass_)
          return false;
      }
    }
  } else if (EventCategory(le, lm, lj, lbj) == 50)
    return false;

  return false;
}

int ttbarEventSelector::EventCategory(int& nEle, int& nMu, int& nJets, int& nbJets) {
  int cat = 0;
  if ((nEle >= 2 || nMu >= 2) && nJets > 1 && nbJets > 1) {  // di-lepton
    if (nEle >= 2 && nJets < 2)
      cat = 11;
    else if (nMu >= 2 && nJets < 2)
      cat = 12;
    else if (nEle >= 1 && nMu >= 1 && nJets < 2)
      cat = 13;
  } else if ((nEle >= 1 || nMu >= 1) && nJets > 3 && nbJets > 2) {  //semi-lepton
    if (nEle >= 1 && nJets > 1)
      cat = 21;
    else if (nMu >= 1 && nJets > 1)
      cat = 22;
  } else if ((nEle < 1 && nMu < 1) && nJets > 5 && nbJets > 1)
    cat = 30;
  else
    cat = 50;
  return cat;
}

double ttbarEventSelector::getMt(const TLorentzVector& vlep, const reco::PFMET& obj) {
  double met = obj.et();
  double phi = obj.phi();

  TLorentzVector vmet;
  double metx = met * std::cos(phi);
  double mety = met * std::sin(phi);
  vmet.SetPxPyPzE(metx, mety, 0.0, met);

  // transverse mass
  TLorentzVector vw = vlep + vmet;

  return std::sqrt(2 * vlep.Et() * met * (1 - std::cos(deltaPhi(vlep.Phi(), phi))));
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ttbarEventSelector);
