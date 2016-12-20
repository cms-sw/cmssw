#ifndef SusyDQM_H
#define SusyDQM_H

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"

class TH1F;
class TH2F;

class PtGreater {
 public:
  template <typename T>
  bool operator()(const T& i, const T& j) {
    return (i.pt() > j.pt());
  }
};

template <typename Mu, typename Ele, typename Jet, typename Met>
class SusyDQM : public DQMEDAnalyzer {
 public:
  explicit SusyDQM(const edm::ParameterSet&);
  ~SusyDQM();

 protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&,
                      edm::EventSetup const&) override;

 private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual bool goodSusyElectron(const Ele*);
  virtual bool goodSusyMuon(const Mu*);

  edm::ParameterSet parameters_;

  std::string moduleName_;

  edm::EDGetTokenT<std::vector<reco::Muon> > muons_;
  edm::EDGetTokenT<std::vector<reco::GsfElectron> > electrons_;
  edm::EDGetTokenT<std::vector<reco::CaloJet> > jets_;
  edm::EDGetTokenT<std::vector<reco::CaloMET> > met_;
  edm::EDGetTokenT<reco::VertexCollection> vertex_;

  double elec_eta_cut_;
  double elec_mva_cut_;
  double elec_d0_cut_;

  double muon_eta_cut_;
  double muon_nHits_cut_;
  double muon_nChi2_cut_;
  double muon_d0_cut_;

  double RAL_muon_pt_cut_;
  double RAL_muon_iso_cut_;

  double RAL_elec_pt_cut_;
  double RAL_elec_iso_cut_;

  double RAL_jet_pt_cut_;
  double RAL_jet_eta_cut_;
  double RAL_jet_min_emf_cut_;
  double RAL_jet_max_emf_cut_;
  double RAL_jet_sum_pt_cut_;

  double RAL_met_cut_;

  math::XYZPoint bs;

  MonitorElement* hRAL_N_muons_;
  MonitorElement* hRAL_pt_muons_;
  MonitorElement* hRAL_eta_muons_;
  MonitorElement* hRAL_phi_muons_;
  MonitorElement* hRAL_Iso_muons_;

  MonitorElement* hRAL_N_elecs_;
  MonitorElement* hRAL_pt_elecs_;
  MonitorElement* hRAL_eta_elecs_;
  MonitorElement* hRAL_phi_elecs_;
  MonitorElement* hRAL_Iso_elecs_;

  MonitorElement* hRAL_Sum_pt_jets_;
  MonitorElement* hRAL_Met_;

  MonitorElement* hRAL_dR_emu_;

  MonitorElement* hRAL_mass_OS_mumu_;
  MonitorElement* hRAL_mass_OS_ee_;
  MonitorElement* hRAL_mass_OS_emu_;
  MonitorElement* hRAL_mass_SS_mumu_;
  MonitorElement* hRAL_mass_SS_ee_;
  MonitorElement* hRAL_mass_SS_emu_;

  MonitorElement* hRAL_Muon_monitor_;
  MonitorElement* hRAL_Electron_monitor_;
  MonitorElement* hRAL_OSee_monitor_;
  MonitorElement* hRAL_OSemu_monitor_;
  MonitorElement* hRAL_OSmumu_monitor_;
  MonitorElement* hRAL_SSee_monitor_;
  MonitorElement* hRAL_SSemu_monitor_;
  MonitorElement* hRAL_SSmumu_monitor_;
  MonitorElement* hRAL_TriMuon_monitor_;
};

template <typename Mu, typename Ele, typename Jet, typename Met>
SusyDQM<Mu, Ele, Jet, Met>::SusyDQM(const edm::ParameterSet& pset) {
  parameters_ = pset;
  moduleName_ = pset.getUntrackedParameter<std::string>("moduleName");

  muons_ = consumes<std::vector<reco::Muon> >(
      pset.getParameter<edm::InputTag>("muonCollection"));
  electrons_ = consumes<std::vector<reco::GsfElectron> >(
      pset.getParameter<edm::InputTag>("electronCollection"));
  jets_ = consumes<std::vector<reco::CaloJet> >(
      pset.getParameter<edm::InputTag>("jetCollection"));
  met_ = consumes<std::vector<reco::CaloMET> >(
      pset.getParameter<edm::InputTag>("metCollection"));
  vertex_ = consumes<reco::VertexCollection>(
      pset.getParameter<edm::InputTag>("vertexCollection"));

  muon_eta_cut_ = pset.getParameter<double>("muon_eta_cut");
  muon_nHits_cut_ = pset.getParameter<double>("muon_nHits_cut");
  muon_nChi2_cut_ = pset.getParameter<double>("muon_nChi2_cut");
  muon_d0_cut_ = pset.getParameter<double>("muon_d0_cut");

  elec_eta_cut_ = pset.getParameter<double>("elec_eta_cut");
  elec_mva_cut_ = pset.getParameter<double>("elec_mva_cut");
  elec_d0_cut_ = pset.getParameter<double>("elec_d0_cut");

  RAL_muon_pt_cut_ = pset.getParameter<double>("RAL_muon_pt_cut");
  RAL_muon_iso_cut_ = pset.getParameter<double>("RAL_muon_iso_cut");

  RAL_elec_pt_cut_ = pset.getParameter<double>("RAL_elec_pt_cut");
  RAL_elec_iso_cut_ = pset.getParameter<double>("RAL_elec_iso_cut");

  RAL_jet_pt_cut_ = pset.getParameter<double>("RAL_jet_pt_cut");
  RAL_jet_sum_pt_cut_ = pset.getParameter<double>("RAL_jet_sum_pt_cut");
  RAL_jet_eta_cut_ = pset.getParameter<double>("RAL_jet_eta_cut");
  RAL_jet_min_emf_cut_ = pset.getParameter<double>("RAL_jet_min_emf_cut");
  RAL_jet_max_emf_cut_ = pset.getParameter<double>("RAL_jet_max_emf_cut");

  RAL_met_cut_ = pset.getParameter<double>("RAL_met_cut");

  hRAL_N_muons_ = 0;
  hRAL_pt_muons_ = 0;
  hRAL_eta_muons_ = 0;
  hRAL_phi_muons_ = 0;
  hRAL_Iso_muons_ = 0;
  hRAL_N_elecs_ = 0;
  hRAL_pt_elecs_ = 0;
  hRAL_eta_elecs_ = 0;
  hRAL_phi_elecs_ = 0;
  hRAL_Iso_elecs_ = 0;
  hRAL_Sum_pt_jets_ = 0;
  hRAL_Met_ = 0;
  hRAL_dR_emu_ = 0;
  hRAL_mass_OS_mumu_ = 0;
  hRAL_mass_OS_ee_ = 0;
  hRAL_mass_OS_emu_ = 0;
  hRAL_mass_SS_mumu_ = 0;
  hRAL_mass_SS_ee_ = 0;
  hRAL_mass_SS_emu_ = 0;
  hRAL_Muon_monitor_ = 0;
  hRAL_Electron_monitor_ = 0;
  hRAL_OSee_monitor_ = 0;
  hRAL_OSemu_monitor_ = 0;
  hRAL_OSmumu_monitor_ = 0;
  hRAL_SSee_monitor_ = 0;
  hRAL_SSemu_monitor_ = 0;
  hRAL_SSmumu_monitor_ = 0;
  hRAL_TriMuon_monitor_ = 0;
}

template <typename Mu, typename Ele, typename Jet, typename Met>
SusyDQM<Mu, Ele, Jet, Met>::~SusyDQM() {}

template <typename Mu, typename Ele, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Jet, Met>::bookHistograms(DQMStore::IBooker& iBooker,
                                                edm::Run const&,
                                                edm::EventSetup const&) {
  iBooker.setCurrentFolder(moduleName_);

  hRAL_N_muons_ = iBooker.book1D("RAL_N_muons", "RAL_N_muons", 10, 0., 10.);
  hRAL_pt_muons_ = iBooker.book1D("RAL_pt_muons", "RAL_pt_muons", 50, 0., 300.);
  hRAL_eta_muons_ =
      iBooker.book1D("RAL_eta_muons", "RAL_eta_muons", 50, -2.5, 2.5);
  hRAL_phi_muons_ = iBooker.book1D("RAL_phi_muons", "RAL_phi_muons", 50, -4., 4.);
  hRAL_Iso_muons_ = iBooker.book1D("RAL_Iso_muons", "RAL_Iso_muons", 50, 0., 25.);

  hRAL_N_elecs_ = iBooker.book1D("RAL_N_elecs", "RAL_N_elecs", 10, 0., 10.);
  hRAL_pt_elecs_ = iBooker.book1D("RAL_pt_elecs", "RAL_pt_elecs", 50, 0., 300.);
  hRAL_eta_elecs_ =
      iBooker.book1D("RAL_eta_elecs", "RAL_eta_elecs", 50, -2.5, 2.5);
  hRAL_phi_elecs_ = iBooker.book1D("RAL_phi_elecs", "RAL_phi_elecs", 50, -4., 4.);
  hRAL_Iso_elecs_ = iBooker.book1D("RAL_Iso_elecs", "RAL_Iso_elecs", 50, 0., 25.);

  hRAL_Sum_pt_jets_ =
      iBooker.book1D("RAL_Sum_pt_jets", "RAL_Sum_pt_jets", 50, 0., 2000.);
  hRAL_Met_ = iBooker.book1D("RAL_Met", "RAL_Met", 50, 0., 1000.);

  hRAL_dR_emu_ = iBooker.book1D("RAL_deltaR_emu", "RAL_deltaR_emu", 50, 0., 10.);

  hRAL_mass_OS_mumu_ =
      iBooker.book1D("RAL_mass_OS_mumu", "RAL_mass_OS_mumu", 50, 0., 300.);
  hRAL_mass_OS_ee_ =
      iBooker.book1D("RAL_mass_OS_ee", "RAL_mass_OS_ee", 50, 0., 300.);
  hRAL_mass_OS_emu_ =
      iBooker.book1D("RAL_mass_OS_emu", "RAL_mass_OS_emu", 50, 0., 300.);
  hRAL_mass_SS_mumu_ =
      iBooker.book1D("RAL_mass_SS_mumu", "RAL_mass_SS_mumu", 50, 0., 300.);
  hRAL_mass_SS_ee_ =
      iBooker.book1D("RAL_mass_SS_ee", "RAL_mass_SS_ee", 50, 0., 300.);
  hRAL_mass_SS_emu_ =
      iBooker.book1D("RAL_mass_SS_emu", "RAL_mass_SS_emu", 50, 0., 300.);

  hRAL_Muon_monitor_ =
      iBooker.book2D("RAL_Single_Muon_Selection", "RAL_Single_Muon_Selection", 50,
                   0., 1000., 50, 0., 1000.);
  hRAL_Electron_monitor_ = iBooker.book2D("RAL_Single_Electron_Selection",
                                        "RAL_Single_Electron_Selection", 50, 0.,
                                        1000., 50, 0., 1000.);
  hRAL_OSee_monitor_ =
      iBooker.book2D("RAL_OS_Electron_Selection", "RAL_OS_Electron_Selection", 50,
                   0., 1000., 50, 0., 1000.);
  hRAL_OSemu_monitor_ = iBooker.book2D("RAL_OS_ElectronMuon_Selection",
                                     "RAL_OS_ElectronMuon_Selection", 50, 0.,
                                     1000., 50, 0., 1000.);
  hRAL_OSmumu_monitor_ =
      iBooker.book2D("RAL_OS_Muon_Selection", "RAL_OS_Muon_Selection", 50, 0.,
                   1000., 50, 0., 1000.);
  hRAL_SSee_monitor_ =
      iBooker.book2D("RAL_SS_Electron_Selection", "RAL_SS_Electron_Selection", 50,
                   0., 1000., 50, 0., 1000.);
  hRAL_SSemu_monitor_ = iBooker.book2D("RAL_SS_ElectronMuon_Selection",
                                     "RAL_SS_ElectronMuon_Selection", 50, 0.,
                                     1000., 50, 0., 1000.);
  hRAL_SSmumu_monitor_ =
      iBooker.book2D("RAL_SS_Muon_Selection", "RAL_SS_Muon_Selection", 50, 0.,
                   1000., 50, 0., 1000.);
  hRAL_TriMuon_monitor_ =
      iBooker.book2D("RAL_Tri_Muon_Selection", "RAL_Tri_Muon_Selection", 50, 0.,
                   1000., 50, 0., 1000.);
}

template <typename Mu, typename Ele, typename Jet, typename Met>
bool SusyDQM<Mu, Ele, Jet, Met>::goodSusyElectron(const Ele* ele) {
  //   if (ele->pt() < elec_pt_cut_)
  //      return false;
  if (fabs(ele->eta()) > elec_eta_cut_) return false;
  //   if (ele->mva() < elec_mva_cut_)
  //      return false;
  if (fabs(ele->gsfTrack()->dxy(bs)) > elec_d0_cut_) return false;
  return true;
}

template <typename Mu, typename Ele, typename Jet, typename Met>
bool SusyDQM<Mu, Ele, Jet, Met>::goodSusyMuon(const Mu* mu) {
  //   if (mu->pt() < muon_pt_cut_)
  //      return false;
  if (fabs(mu->eta()) > muon_eta_cut_) return false;
  if (!mu->isGlobalMuon()) return false;
  if (mu->innerTrack()->numberOfValidHits() < muon_nHits_cut_) return false;
  if (mu->globalTrack()->normalizedChi2() > muon_nChi2_cut_) return false;
  if (fabs(mu->innerTrack()->dxy(bs)) > muon_d0_cut_) return false;
  return true;
}

template <typename Mu, typename Ele, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Jet, Met>::analyze(const edm::Event& evt,
                                         const edm::EventSetup& iSetup) {
  edm::Handle<std::vector<Mu> > muons;
  bool isFound = evt.getByToken(muons_, muons);
  if (!isFound) return;

  edm::Handle<std::vector<Ele> > elecs;
  isFound = evt.getByToken(electrons_, elecs);
  if (!isFound) return;

  //// sorted jets
  edm::Handle<std::vector<Jet> > cJets;
  isFound = evt.getByToken(jets_, cJets);
  if (!isFound) return;
  std::vector<Jet> jets = *cJets;
  std::sort(jets.begin(), jets.end(), PtGreater());

  edm::Handle<std::vector<Met> > mets;
  isFound = evt.getByToken(met_, mets);
  if (!isFound) return;

  edm::Handle<reco::VertexCollection> vertices;
  isFound = evt.getByToken(vertex_, vertices);
  if (!isFound) return;

  //////////////////////////////
  // Leptonic DQM histos
  //////////////////////////////

  float sumPt = 0.;
  for (typename std::vector<Jet>::const_iterator jet_i = jets.begin();
       jet_i != jets.end(); ++jet_i) {
    if (jet_i->pt() < RAL_jet_pt_cut_) continue;
    if (fabs(jet_i->eta()) > RAL_jet_eta_cut_) continue;
    if (fabs(jet_i->eta()) > RAL_jet_eta_cut_) continue;
    if (jet_i->emEnergyFraction() < RAL_jet_min_emf_cut_) continue;
    if (jet_i->emEnergyFraction() > RAL_jet_max_emf_cut_) continue;
    sumPt += jet_i->pt();
  }

  hRAL_Sum_pt_jets_->Fill(sumPt);

  float MET = 0.;
  for (typename std::vector<Met>::const_iterator met_i = mets->begin();
       met_i != mets->end(); ++met_i) {
    MET = met_i->pt();
    break;
  }

  hRAL_Met_->Fill(MET);

  int nMuons = 0;
  int nSSmumu = 0;
  int nOSmumu = 0;
  int nSSemu = 0;
  int nOSemu = 0;
  float inv = 0.;
  float dR = 0.;

  for (typename std::vector<Mu>::const_iterator mu_i = muons->begin();
       mu_i != muons->end(); ++mu_i) {
    if (!(goodSusyMuon(&(*mu_i)) && mu_i->pt() > RAL_muon_pt_cut_)) continue;
    ++nMuons;

    hRAL_pt_muons_->Fill(mu_i->pt());
    hRAL_eta_muons_->Fill(mu_i->eta());
    hRAL_phi_muons_->Fill(mu_i->phi());

    reco::MuonIsolation muIso = mu_i->isolationR03();
    hRAL_Iso_muons_->Fill(muIso.emEt + muIso.hadEt + muIso.sumPt);

    // Muon muon pairs
    for (typename std::vector<Mu>::const_iterator mu_j = muons->begin();
         mu_j != muons->end(); ++mu_j) {
      if (mu_i >= mu_j) continue;
      if (!(goodSusyMuon(&(*mu_j)) && mu_j->pt() > RAL_muon_pt_cut_)) continue;

      inv = (mu_i->p4() + mu_j->p4()).M();
      if (mu_i->charge() * mu_j->charge() > 0) {
        ++nSSmumu;
        hRAL_mass_SS_mumu_->Fill(inv);
      }
      if (mu_i->charge() * mu_j->charge() < 0) {
        ++nOSmumu;
        hRAL_mass_OS_mumu_->Fill(inv);
      }
    }

    // Electron muon pairs
    for (typename std::vector<Ele>::const_iterator ele_j = elecs->begin();
         ele_j != elecs->end(); ++ele_j) {
      if (!(goodSusyElectron(&(*ele_j)) && ele_j->pt() > RAL_elec_pt_cut_))
        continue;
      inv = (mu_i->p4() + ele_j->p4()).M();
      dR = deltaR(*mu_i, *ele_j);
      hRAL_dR_emu_->Fill(dR);
      if (mu_i->charge() * ele_j->charge() > 0) {
        ++nSSemu;
        hRAL_mass_SS_emu_->Fill(inv);
      }
      if (mu_i->charge() * ele_j->charge() < 0) {
        ++nOSemu;
        hRAL_mass_OS_emu_->Fill(inv);
      }
    }
  }

  hRAL_N_muons_->Fill(nMuons);

  int nElectrons = 0;
  int nSSee = 0;
  int nOSee = 0;
  for (typename std::vector<Ele>::const_iterator ele_i = elecs->begin();
       ele_i != elecs->end(); ++ele_i) {
    if (!(goodSusyElectron(&(*ele_i)) && ele_i->pt() > RAL_elec_pt_cut_))
      continue;
    nElectrons++;

    hRAL_pt_elecs_->Fill(ele_i->pt());
    hRAL_eta_elecs_->Fill(ele_i->eta());
    hRAL_phi_elecs_->Fill(ele_i->phi());

    hRAL_Iso_elecs_->Fill(ele_i->dr03TkSumPt() + ele_i->dr03EcalRecHitSumEt() +
                          ele_i->dr03HcalTowerSumEt());

    // Electron electron pairs
    for (typename std::vector<Ele>::const_iterator ele_j = elecs->begin();
         ele_j != elecs->end(); ++ele_j) {
      if (ele_i >= ele_j) continue;
      if (!(goodSusyElectron(&(*ele_j)) && ele_j->pt() > RAL_elec_pt_cut_))
        continue;

      inv = (ele_i->p4() + ele_j->p4()).M();
      if (ele_i->charge() * ele_j->charge() > 0) {
        ++nSSee;
        hRAL_mass_SS_ee_->Fill(inv);
      }
      if (ele_i->charge() * ele_j->charge() < 0) {
        ++nOSee;
        hRAL_mass_OS_ee_->Fill(inv);
      }
    }
  }

  hRAL_N_elecs_->Fill(nElectrons);

  if (MET > RAL_met_cut_ && sumPt > RAL_jet_sum_pt_cut_) {
    if (nMuons >= 1) {
      hRAL_Muon_monitor_->Fill(sumPt, MET);
    }
    if (nElectrons >= 1) {
      hRAL_Electron_monitor_->Fill(sumPt, MET);
    }
    if (nOSee >= 1) {
      hRAL_OSee_monitor_->Fill(sumPt, MET);
    }
    if (nOSemu >= 1) {
      hRAL_OSemu_monitor_->Fill(sumPt, MET);
    }
    if (nOSmumu >= 1) {
      hRAL_OSmumu_monitor_->Fill(sumPt, MET);
    }
    if (nSSee >= 1) {
      hRAL_SSee_monitor_->Fill(sumPt, MET);
    }
    if (nSSemu >= 1) {
      hRAL_SSemu_monitor_->Fill(sumPt, MET);
    }
    if (nSSmumu >= 1) {
      hRAL_SSmumu_monitor_->Fill(sumPt, MET);
    }
  }
  if (nMuons >= 3) {
    hRAL_TriMuon_monitor_->Fill(sumPt, MET);
  }
}

#endif

typedef SusyDQM<reco::Muon, reco::GsfElectron, reco::CaloJet, reco::CaloMET>
    RecoSusyDQM;
