// W->lnu Event Selector

// user includes
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// ROOT includes
#include "TLorentzVector.h"
#include "TMath.h"

class WtoLNuSelector : public edm::stream::EDFilter<> {
public:
  explicit WtoLNuSelector(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::Event&, edm::EventSetup const&) override;
  double getMt(const TLorentzVector& vlep, const reco::PFMET& obj);

private:
  // module config parameters
  const edm::InputTag electronTag_;
  const edm::InputTag bsTag_;
  const edm::InputTag muonTag_;
  const edm::InputTag pfmetTag_;
  const edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  const edm::EDGetTokenT<reco::PFMETCollection> pfmetToken_;
};

void WtoLNuSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("electronInputTag", edm::InputTag("gedGsfElectrons"));
  desc.addUntracked<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"));
  desc.addUntracked<edm::InputTag>("muonInputTag", edm::InputTag("muons"));
  desc.addUntracked<edm::InputTag>("pfmetTag", edm::InputTag("pfMetT1T2Txy"));
  descriptions.addWithDefaultLabel(desc);
}

WtoLNuSelector::WtoLNuSelector(const edm::ParameterSet& ps)
    : electronTag_(ps.getUntrackedParameter<edm::InputTag>("electronInputTag", edm::InputTag("gedGsfElectrons"))),
      bsTag_(ps.getUntrackedParameter<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"))),
      muonTag_(ps.getUntrackedParameter<edm::InputTag>("muonInputTag", edm::InputTag("muons"))),
      pfmetTag_(ps.getUntrackedParameter<edm::InputTag>("pfmetTag", edm::InputTag("pfMetT1T2Txy"))),
      electronToken_(consumes<reco::GsfElectronCollection>(electronTag_)),
      bsToken_(consumes<reco::BeamSpot>(bsTag_)),
      muonToken_(consumes<reco::MuonCollection>(muonTag_)),
      pfmetToken_(consumes<reco::PFMETCollection>(pfmetTag_)) {}

bool WtoLNuSelector::filter(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // Read Electron Collection
  edm::Handle<reco::GsfElectronCollection> electronColl;
  iEvent.getByToken(electronToken_, electronColl);

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(bsToken_, beamSpot);

  std::vector<TLorentzVector> eleList;
  if (electronColl.isValid()) {
    for (auto const& ele : *electronColl) {
      if (!ele.ecalDriven())
        continue;

      double hOverE = ele.hadronicOverEm();
      double sigmaee = ele.sigmaIetaIeta();
      double deltaPhiIn = ele.deltaPhiSuperClusterTrackAtVtx();
      double deltaEtaIn = ele.deltaEtaSuperClusterTrackAtVtx();

      // separate cut for barrel and endcap
      if (ele.isEB()) {
        if (std::fabs(deltaPhiIn) >= .15 && std::fabs(deltaEtaIn) >= .007 && hOverE >= .12 && sigmaee >= .01)
          continue;
      } else if (ele.isEE()) {
        if (std::fabs(deltaPhiIn) >= .10 && std::fabs(deltaEtaIn) >= .009 && hOverE >= .10 && sigmaee >= .03)
          continue;
      }

      reco::GsfTrackRef trk = ele.gsfTrack();
      if (!trk.isNonnull())
        continue;  // only electrons wd tracks
      double chi2 = trk->chi2();
      double ndof = trk->ndof();
      double chbyndof = (ndof > 0) ? chi2 / ndof : 0;
      double trkd0 = trk->d0();
      double trkdz = trk->dz();
      if (beamSpot.isValid()) {
        trkd0 = -(trk->dxy(beamSpot->position()));
        trkdz = trk->dz(beamSpot->position());
      } else {
        edm::LogError("WtoLNuSelector") << "Error >> Failed to get BeamSpot for label: " << bsTag_;
      }
      if (chbyndof >= 10 || std::fabs(trkd0) >= 0.02 || std::fabs(trkdz) >= 20)
        continue;
      const reco::HitPattern& hitp = trk->hitPattern();
      int nPixelHits = hitp.numberOfValidPixelHits();
      int nStripHits = hitp.numberOfValidStripHits();
      if (nPixelHits < 1 || nStripHits < 8)
        continue;

      // PF Isolation
      reco::GsfElectron::PflowIsolationVariables pfIso = ele.pfIsolationVariables();
      float absiso =
          pfIso.sumChargedHadronPt + std::max(0.0, pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt - 0.5 * pfIso.sumPUPt);
      float eiso = absiso / ele.pt();
      if (eiso > 0.3)
        continue;

      TLorentzVector le;
      le.SetPtEtaPhiE(ele.pt(), ele.eta(), ele.phi(), ele.energy());
      eleList.push_back(le);
    }
  } else {
    edm::LogError("WtoLNuSelector") << "Error >> Failed to get ElectronCollection for label: " << electronTag_;
  }

  // Read Muon Collection
  edm::Handle<reco::MuonCollection> muonColl;
  iEvent.getByToken(muonToken_, muonColl);

  std::vector<TLorentzVector> muList;
  if (muonColl.isValid()) {
    for (auto const& mu : *muonColl) {
      if (!mu.isGlobalMuon() || !mu.isPFMuon() || std::fabs(mu.eta()) > 2.1 || mu.pt() <= 5)
        continue;

      reco::TrackRef gtrkref = mu.globalTrack();
      if (!gtrkref.isNonnull())
        continue;
      const reco::Track* gtk = &(*gtrkref);
      double chi2 = gtk->chi2();
      double ndof = gtk->ndof();
      double chbyndof = (ndof > 0) ? chi2 / ndof : 0;

      const reco::HitPattern& hitp = gtk->hitPattern();
      int nPixelHits = hitp.numberOfValidPixelHits();
      int nStripHits = hitp.numberOfValidStripHits();

      reco::TrackRef itrkref = mu.innerTrack();  // tracker segment only
      if (!itrkref.isNonnull())
        continue;
      const reco::Track* tk = &(*itrkref);
      double trkd0 = tk->d0();
      double trkdz = tk->dz();
      if (beamSpot.isValid()) {
        trkd0 = -(tk->dxy(beamSpot->position()));
        trkdz = tk->dz(beamSpot->position());
      }
      // Hits/section in the muon chamber
      int nChambers = mu.numberOfChambers();
      int nMatches = mu.numberOfMatches();
      int nMatchedStations = mu.numberOfMatchedStations();

      // PF Isolation
      const reco::MuonPFIsolation& pfIso04 = mu.pfIsolationR04();
      double absiso = pfIso04.sumChargedParticlePt +
                      std::max(0.0, pfIso04.sumNeutralHadronEt + pfIso04.sumPhotonEt - 0.5 * pfIso04.sumPUPt);

      // Final selection
      if (chbyndof < 10 && std::fabs(trkd0) < 0.02 && std::fabs(trkdz) < 20.0 && nPixelHits > 1 && nStripHits > 8 &&
          nChambers > 2 && nMatches > 2 && nMatchedStations > 2 && absiso / mu.pt() < 0.3) {
        TLorentzVector lm;
        lm.SetPtEtaPhiE(mu.pt(), mu.eta(), mu.phi(), mu.energy());
        muList.push_back(lm);
      }
    }
  } else {
    edm::LogError("WtoLNuSelector") << "Error >> Failed to get MuonCollection for label: " << muonTag_;
  }

  // Require either a high pt electron or muon
  if (eleList.empty() && muList.empty())
    return false;

  // Both should not be present at the same time
  if ((!eleList.empty() && eleList[0].Pt() > 20) && (!muList.empty() && muList[0].Pt() > 20))
    return false;

  // find the high pt lepton
  TLorentzVector vlep;
  if (!eleList.empty() && !muList.empty()) {
    vlep = (eleList[0].Pt() > muList[0].Pt()) ? eleList[0] : muList[0];
  } else if (!eleList.empty()) {
    vlep = eleList[0];
  } else {
    vlep = muList[0];
  }
  if (vlep.Pt() < 20)
    return false;

  edm::Handle<reco::PFMETCollection> pfColl;
  iEvent.getByToken(pfmetToken_, pfColl);

  if (pfColl.isValid()) {
    double mt = getMt(vlep, pfColl->front());
    if (mt < 60 || mt > 80)
      return false;
  } else {
    edm::LogError("WtoLNuSelector") << "Error >> Failed to get PFMETCollection for label: " << pfmetTag_;
    return false;
  }

  return true;
}
double WtoLNuSelector::getMt(const TLorentzVector& vlep, const reco::PFMET& obj) {
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
DEFINE_FWK_MODULE(WtoLNuSelector);
