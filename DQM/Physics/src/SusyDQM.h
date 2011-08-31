#ifndef SusyDQM_H
#define SusyDQM_H

#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"

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

//#include "DataFormats/PatCandidates/interface/Electron.h"
//#include "DataFormats/PatCandidates/interface/Muon.h"
//#include "DataFormats/PatCandidates/interface/Jet.h"
//#include "DataFormats/PatCandidates/interface/MET.h"

class TH1F;
class TH2F;

class PtGreater {
   public:
      template<typename T> bool operator ()(const T& i, const T& j) {
         return (i.pt() > j.pt());
      }
};

template<typename Mu, typename Ele, typename Jet, typename Met>
class SusyDQM: public edm::EDAnalyzer {

   public:

      explicit SusyDQM(const edm::ParameterSet&);
      ~SusyDQM();

   protected:

      void beginRun(const edm::Run&);
      void endRun(const edm::Run&);

   private:

      void initialize();
      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual bool goodSusyElectron(const Ele*);
      virtual bool goodSusyMuon(const Mu*);
      virtual void endJob();

      edm::ParameterSet parameters_;
      DQMStore * dbe_;

      std::string moduleName_;

      edm::InputTag muons_;
      edm::InputTag electrons_;
      edm::InputTag jets_;
      edm::InputTag met_;
      edm::InputTag vertex_;

      double elec_eta_cut_;
      double elec_mva_cut_;
      double elec_d0_cut_;

      double muon_eta_cut_;
      double muon_nHits_cut_;
      double muon_nChi2_cut_;
      double muon_d0_cut_;

      double RA12_muon_pt_cut_;
      double RA12_muon_iso_cut_;

      double RA12_elec_pt_cut_;
      double RA12_elec_iso_cut_;

      double RA1_jet_pt_cut_;
      double RA1_jet_eta_cut_;
      double RA1_jet_min_emf_cut_;
      double RA1_jet_max_emf_cut_;
      double RA1_jet1_pt_cut_;
      double RA1_jet1_eta_cut_;
      double RA1_jet2_pt_cut_;
      double RA1_jet2_eta_cut_;
      double RA1_jet3_pt_cut_;

      double RA1_alphat_cut_;
      double RA1_ht_cut_;
      double RA1_mht_cut_;
      double RA1_deltaPhi_cut_;
      double RA1_deltaPhiJets_cut_;

      double RA2_jet_pt_cut_;
      double RA2_jet_eta_cut_;
      double RA2_jet_min_emf_cut_;
      double RA2_jet_max_emf_cut_;
      double RA2_jet1_pt_cut_;
      double RA2_jet2_pt_cut_;
      double RA2_jet3_pt_cut_;
      double RA2_jet1_eta_cut_;
      double RA2_jet2_eta_cut_;
      double RA2_jet3_eta_cut_;
      int RA2_N_jets_cut_;

      double RA2_ht_cut_;
      double RA2_mht_cut_;
      double RA2_deltaPhi_cut_;

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

      MonitorElement * hRA1_pt_jet1_nm1_;
      MonitorElement * hRA1_eta_jet1_nm1_;
      MonitorElement * hRA1_pt_jet2_nm1_;
      MonitorElement * hRA1_eta_jet2_nm1_;
      MonitorElement * hRA1_pt_jet3_nm1_;
      MonitorElement * hRA1_deltaPhi_mht_jets_nm1_;
      MonitorElement * hRA1_deltaPhi_jets_nm1_;
      MonitorElement * hRA1_ht_nm1_;
      MonitorElement * hRA1_mht_nm1_;
      MonitorElement * hRA1_alphat_nm1_;
      MonitorElement * hRA1_pt_muons_nm1_;
      MonitorElement * hRA1_pt_elecs_nm1_;

      MonitorElement * hRA2_N_jets_nm1_;
      MonitorElement * hRA2_pt_jet1_nm1_;
      MonitorElement * hRA2_eta_jet1_nm1_;
      MonitorElement * hRA2_pt_jet2_nm1_;
      MonitorElement * hRA2_eta_jet2_nm1_;
      MonitorElement * hRA2_pt_jet3_nm1_;
      MonitorElement * hRA2_eta_jet3_nm1_;
      MonitorElement * hRA2_deltaPhi_mht_jets_nm1_;
      MonitorElement * hRA2_ht_nm1_;
      MonitorElement * hRA2_mht_nm1_;
      MonitorElement * hRA2_pt_muons_nm1_;
      MonitorElement * hRA2_pt_elecs_nm1_;

      MonitorElement * hRA1_pt_jet1_;
      MonitorElement * hRA1_eta_jet1_;
      MonitorElement * hRA1_pt_jet2_;
      MonitorElement * hRA1_eta_jet2_;
      MonitorElement * hRA1_pt_jet3_;
      MonitorElement * hRA1_deltaPhi_mht_jets_;
      MonitorElement * hRA1_deltaPhi_jets_;
      MonitorElement * hRA1_ht_;
      MonitorElement * hRA1_mht_;
      MonitorElement * hRA1_alphat_;
      MonitorElement * hRA1_pt_muons_;
      MonitorElement * hRA1_pt_elecs_;

      MonitorElement * hRA2_N_jets_;
      MonitorElement * hRA2_pt_jet1_;
      MonitorElement * hRA2_eta_jet1_;
      MonitorElement * hRA2_pt_jet2_;
      MonitorElement * hRA2_eta_jet2_;
      MonitorElement * hRA2_pt_jet3_;
      MonitorElement * hRA2_eta_jet3_;
      MonitorElement * hRA2_deltaPhi_mht_jets_;
      MonitorElement * hRA2_ht_;
      MonitorElement * hRA2_mht_;
      MonitorElement * hRA2_pt_muons_;
      MonitorElement * hRA2_pt_elecs_;

      MonitorElement * hRAL_N_muons_;
      MonitorElement * hRAL_pt_muons_;
      MonitorElement * hRAL_eta_muons_;
      MonitorElement * hRAL_phi_muons_;
      MonitorElement * hRAL_Iso_muons_;

      MonitorElement * hRAL_N_elecs_;
      MonitorElement * hRAL_pt_elecs_;
      MonitorElement * hRAL_eta_elecs_;
      MonitorElement * hRAL_phi_elecs_;
      MonitorElement * hRAL_Iso_elecs_;

      MonitorElement * hRAL_Sum_pt_jets_;
      MonitorElement * hRAL_Met_;

      MonitorElement * hRAL_dR_emu_;

      MonitorElement * hRAL_mass_OS_mumu_;
      MonitorElement * hRAL_mass_OS_ee_;
      MonitorElement * hRAL_mass_OS_emu_;
      MonitorElement * hRAL_mass_SS_mumu_;
      MonitorElement * hRAL_mass_SS_ee_;
      MonitorElement * hRAL_mass_SS_emu_;

      MonitorElement * hRAL_Muon_monitor_;
      MonitorElement * hRAL_Electron_monitor_;
      MonitorElement * hRAL_OSee_monitor_;
      MonitorElement * hRAL_OSemu_monitor_;
      MonitorElement * hRAL_OSmumu_monitor_;
      MonitorElement * hRAL_SSee_monitor_;
      MonitorElement * hRAL_SSemu_monitor_;
      MonitorElement * hRAL_SSmumu_monitor_;
      MonitorElement * hRAL_TriMuon_monitor_;

};

template<typename Mu, typename Ele, typename Jet, typename Met>
SusyDQM<Mu, Ele, Jet, Met>::SusyDQM(const edm::ParameterSet& pset) {

   parameters_ = pset;
   initialize();

   moduleName_ = pset.getUntrackedParameter<std::string> ("moduleName");

   muons_ = pset.getParameter<edm::InputTag> ("muonCollection");
   electrons_ = pset.getParameter<edm::InputTag> ("electronCollection");
   jets_ = pset.getParameter<edm::InputTag> ("jetCollection");
   met_ = pset.getParameter<edm::InputTag> ("metCollection");
   vertex_ = pset.getParameter<edm::InputTag> ("vertexCollection");

   muon_eta_cut_ = pset.getParameter<double> ("muon_eta_cut");
   muon_nHits_cut_ = pset.getParameter<double> ("muon_nHits_cut");
   muon_nChi2_cut_ = pset.getParameter<double> ("muon_nChi2_cut");
   muon_d0_cut_ = pset.getParameter<double> ("muon_d0_cut");

   elec_eta_cut_ = pset.getParameter<double> ("elec_eta_cut");
   elec_mva_cut_ = pset.getParameter<double> ("elec_mva_cut");
   elec_d0_cut_ = pset.getParameter<double> ("elec_d0_cut");

   RA12_muon_pt_cut_ = pset.getParameter<double> ("RA12_muon_pt_cut");
   RA12_muon_iso_cut_ = pset.getParameter<double> ("RA12_muon_iso_cut");

   RA12_elec_pt_cut_ = pset.getParameter<double> ("RA12_elec_pt_cut");
   RA12_elec_iso_cut_ = pset.getParameter<double> ("RA12_elec_iso_cut");

   RA1_jet_pt_cut_ = pset.getParameter<double> ("RA1_jet_pt_cut");
   RA1_jet_eta_cut_ = pset.getParameter<double> ("RA1_jet_eta_cut");
   RA1_jet_min_emf_cut_ = pset.getParameter<double> ("RA1_jet_min_emf_cut");
   RA1_jet_max_emf_cut_ = pset.getParameter<double> ("RA1_jet_max_emf_cut");
   RA1_jet1_pt_cut_ = pset.getParameter<double> ("RA1_jet1_pt_cut");
   RA1_jet1_eta_cut_ = pset.getParameter<double> ("RA1_jet1_eta_cut");
   RA1_jet2_pt_cut_ = pset.getParameter<double> ("RA1_jet2_pt_cut");
   RA1_jet2_eta_cut_ = pset.getParameter<double> ("RA1_jet2_eta_cut");
   RA1_jet3_pt_cut_ = pset.getParameter<double> ("RA1_jet3_pt_cut");

   RA1_alphat_cut_ = pset.getParameter<double> ("RA1_alphat_cut");
   RA1_ht_cut_ = pset.getParameter<double> ("RA1_ht_cut");
   RA1_mht_cut_ = pset.getParameter<double> ("RA1_mht_cut");
   RA1_deltaPhi_cut_ = pset.getParameter<double> ("RA1_deltaPhi_cut");
   RA1_deltaPhiJets_cut_ = pset.getParameter<double> ("RA1_deltaPhiJets_cut");

   RA2_jet_pt_cut_ = pset.getParameter<double> ("RA2_jet_pt_cut");
   RA2_jet_eta_cut_ = pset.getParameter<double> ("RA2_jet_eta_cut");
   RA2_jet_min_emf_cut_ = pset.getParameter<double> ("RA2_jet_min_emf_cut");
   RA2_jet_max_emf_cut_ = pset.getParameter<double> ("RA2_jet_max_emf_cut");
   RA2_jet1_pt_cut_ = pset.getParameter<double> ("RA2_jet1_pt_cut");
   RA2_jet1_eta_cut_ = pset.getParameter<double> ("RA2_jet1_eta_cut");
   RA2_jet2_pt_cut_ = pset.getParameter<double> ("RA2_jet2_pt_cut");
   RA2_jet2_eta_cut_ = pset.getParameter<double> ("RA2_jet2_eta_cut");
   RA2_jet3_pt_cut_ = pset.getParameter<double> ("RA2_jet3_pt_cut");
   RA2_jet3_eta_cut_ = pset.getParameter<double> ("RA2_jet3_eta_cut");
   RA2_N_jets_cut_ = pset.getParameter<int> ("RA2_N_jets_cut");

   RA2_ht_cut_ = pset.getParameter<double> ("RA2_ht_cut");
   RA2_mht_cut_ = pset.getParameter<double> ("RA2_mht_cut");
   RA2_deltaPhi_cut_ = pset.getParameter<double> ("RA2_deltaPhi_cut");

   RAL_muon_pt_cut_ = pset.getParameter<double> ("RAL_muon_pt_cut");
   RAL_muon_iso_cut_ = pset.getParameter<double> ("RAL_muon_iso_cut");

   RAL_elec_pt_cut_ = pset.getParameter<double> ("RAL_elec_pt_cut");
   RAL_elec_iso_cut_ = pset.getParameter<double> ("RAL_elec_iso_cut");

   RAL_jet_pt_cut_ = pset.getParameter<double> ("RAL_jet_pt_cut");
   RAL_jet_sum_pt_cut_ = pset.getParameter<double> ("RAL_jet_sum_pt_cut");
   RAL_jet_eta_cut_ = pset.getParameter<double> ("RAL_jet_eta_cut");
   RAL_jet_min_emf_cut_ = pset.getParameter<double> ("RAL_jet_min_emf_cut");
   RAL_jet_max_emf_cut_ = pset.getParameter<double> ("RAL_jet_max_emf_cut");

   RAL_met_cut_ = pset.getParameter<double> ("RAL_met_cut");
}

template<typename Mu, typename Ele, typename Jet, typename Met>
SusyDQM<Mu, Ele, Jet, Met>::~SusyDQM() {

}

template<typename Mu, typename Ele, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Jet, Met>::initialize() {

}

template<typename Mu, typename Ele, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Jet, Met>::beginJob() {

   dbe_ = edm::Service<DQMStore>().operator->();

   dbe_->setCurrentFolder(moduleName_);

   hRA1_pt_jet1_nm1_ = dbe_->book1D("RA1_pt_jet1_nm1", "RA1_pt_jet1_nm1", 50, 0., 1000.);
   hRA1_eta_jet1_nm1_ = dbe_->book1D("RA1_eta_jet1_nm1", "RA1_eta_jet1_nm1", 50, -5., 5.);
   hRA1_pt_jet2_nm1_ = dbe_->book1D("RA1_pt_jet2_nm1", "RA1_pt_jet2_nm1", 50, 0., 1000.);
   hRA1_eta_jet2_nm1_ = dbe_->book1D("RA1_eta_jet2_nm1", "RA1_eta_jet2_nm1", 50, -5., 5.);
   hRA1_pt_jet3_nm1_ = dbe_->book1D("RA1_pt_jet3_nm1", "RA1_pt_jet3_nm1", 50, 0., 1000.);
   hRA1_deltaPhi_mht_jets_nm1_ = dbe_->book1D("RA1_deltaPhi_mht_jets_nm1", "RA1_deltaPhi_mht_jets_nm1", 50, 0., 2.);
   hRA1_deltaPhi_jets_nm1_ = dbe_->book1D("RA1_deltaPhi_jets_nm1", "RA1_deltaPhi_jets_nm1", 50, 0., 4.);
   hRA1_ht_nm1_ = dbe_->book1D("RA1_ht_nm1", "RA1_ht_nm1", 50, 0., 1000.);
   hRA1_mht_nm1_ = dbe_->book1D("RA1_mht_nm1", "RA1_mht_nm1", 50, 0., 1000.);
   hRA1_alphat_nm1_ = dbe_->book1D("RA1_alphat_nm1", "RA1_alphat_nm1", 50, 0., 2.);
   hRA1_pt_muons_nm1_ = dbe_->book1D("RA1_pt_muons_nm1", "RA1_pt_muons_nm1", 50, 0., 200.);
   hRA1_pt_elecs_nm1_ = dbe_->book1D("RA1_pt_elecs_nm1", "RA1_pt_elecs_nm1", 50, 0., 200.);

   hRA2_N_jets_nm1_ = dbe_->book1D("RA2_N_jets_nm1", "RA2_N_jets_nm1", 10, 0., 10.);
   hRA2_pt_jet1_nm1_ = dbe_->book1D("RA2_pt_jet1_nm1", "RA2_pt_jet1_nm1", 50, 0., 1000.);
   hRA2_eta_jet1_nm1_ = dbe_->book1D("RA2_eta_jet1_nm1", "RA2_eta_jet1_nm1", 50, -5., 5.);
   hRA2_pt_jet2_nm1_ = dbe_->book1D("RA2_pt_jet2_nm1", "RA2_pt_jet2_nm1", 50, 0., 1000.);
   hRA2_eta_jet2_nm1_ = dbe_->book1D("RA2_eta_jet2_nm1", "RA2_eta_jet2_nm1", 50, -5., 5.);
   hRA2_pt_jet3_nm1_ = dbe_->book1D("RA2_pt_jet3_nm1", "RA2_pt_jet3_nm1", 50, 0., 1000.);
   hRA2_eta_jet3_nm1_ = dbe_->book1D("RA2_eta_jet3_nm1", "RA2_eta_jet3_nm1", 50, -5., 5.);
   hRA2_deltaPhi_mht_jets_nm1_ = dbe_->book1D("RA2_deltaPhi_mht_jets_nm1", "RA2_deltaPhi_mht_jets_nm1", 50, 0., 2.);
   hRA2_ht_nm1_ = dbe_->book1D("RA2_ht_nm1", "RA2_ht_nm1", 50, 0., 2000.);
   hRA2_mht_nm1_ = dbe_->book1D("RA2_mht_nm1", "RA2_mht_nm1", 50, 0., 1000.);
   hRA2_pt_muons_nm1_ = dbe_->book1D("RA2_pt_muons_nm1", "RA2_pt_muons_nm1", 50, 0., 200.);
   hRA2_pt_elecs_nm1_ = dbe_->book1D("RA2_pt_elecs_nm1", "RA2_pt_elecs_nm1", 50, 0., 200.);

   hRA1_pt_jet1_ = dbe_->book1D("RA1_pt_jet1", "RA1_pt_jet1", 50, 0., 1000.);
   hRA1_eta_jet1_ = dbe_->book1D("RA1_eta_jet1", "RA1_eta_jet1", 50, -5., 5.);
   hRA1_pt_jet2_ = dbe_->book1D("RA1_pt_jet2", "RA1_pt_jet2", 50, 0., 1000.);
   hRA1_eta_jet2_ = dbe_->book1D("RA1_eta_jet2", "RA1_eta_jet2", 50, -5., 5.);
   hRA1_pt_jet3_ = dbe_->book1D("RA1_pt_jet3", "RA1_pt_jet3", 50, 0., 1000.);
   hRA1_deltaPhi_mht_jets_ = dbe_->book1D("RA1_deltaPhi_mht_jets", "RA1_deltaPhi_mht_jets", 50, 0., 2.);
   hRA1_deltaPhi_jets_ = dbe_->book1D("RA1_deltaPhi_jets", "RA1_deltaPhi_jets", 50, 0., 4.);
   hRA1_ht_ = dbe_->book1D("RA1_ht", "RA1_ht", 50, 0., 1000.);
   hRA1_mht_ = dbe_->book1D("RA1_mht", "RA1_mht", 50, 0., 1000.);
   hRA1_alphat_ = dbe_->book1D("RA1_alphat", "RA1_alphat", 50, 0., 2.);
   hRA1_pt_muons_ = dbe_->book1D("RA1_pt_muons", "RA1_pt_muons", 50, 0., 200.);
   hRA1_pt_elecs_ = dbe_->book1D("RA1_pt_elecs", "RA1_pt_elecs", 50, 0., 200.);

   hRA2_N_jets_ = dbe_->book1D("RA2_N_jets", "RA2_N_jets", 10, 0., 10.);
   hRA2_pt_jet1_ = dbe_->book1D("RA2_pt_jet1", "RA2_pt_jet1", 50, 0., 1000.);
   hRA2_eta_jet1_ = dbe_->book1D("RA2_eta_jet1", "RA2_eta_jet1", 50, -5., 5.);
   hRA2_pt_jet2_ = dbe_->book1D("RA2_pt_jet2", "RA2_pt_jet2", 50, 0., 1000.);
   hRA2_eta_jet2_ = dbe_->book1D("RA2_eta_jet2", "RA2_eta_jet2", 50, -5., 5.);
   hRA2_pt_jet3_ = dbe_->book1D("RA2_pt_jet3", "RA2_pt_jet3", 50, 0., 1000.);
   hRA2_eta_jet3_ = dbe_->book1D("RA2_eta_jet3", "RA2_eta_jet3", 50, -5., 5.);
   hRA2_deltaPhi_mht_jets_ = dbe_->book1D("RA2_deltaPhi_mht_jets", "RA2_deltaPhi_mht_jets", 50, 0., 2.);
   hRA2_ht_ = dbe_->book1D("RA2_ht", "RA2_ht", 50, 0., 2000.);
   hRA2_mht_ = dbe_->book1D("RA2_mht", "RA2_mht", 50, 0., 1000.);
   hRA2_pt_muons_ = dbe_->book1D("RA2_pt_muons", "RA2_pt_muons", 50, 0., 200.);
   hRA2_pt_elecs_ = dbe_->book1D("RA2_pt_elecs", "RA2_pt_elecs", 50, 0., 200.);

   hRAL_N_muons_ = dbe_->book1D("RAL_N_muons", "RAL_N_muons", 10, 0., 10.);
   hRAL_pt_muons_ = dbe_->book1D("RAL_pt_muons", "RAL_pt_muons", 50, 0., 300.);
   hRAL_eta_muons_ = dbe_->book1D("RAL_eta_muons", "RAL_eta_muons", 50, -2.5, 2.5);
   hRAL_phi_muons_ = dbe_->book1D("RAL_phi_muons", "RAL_phi_muons", 50, -4., 4.);
   hRAL_Iso_muons_ = dbe_->book1D("RAL_Iso_muons", "RAL_Iso_muons", 50, 0., 25.);

   hRAL_N_elecs_ = dbe_->book1D("RAL_N_elecs", "RAL_N_elecs", 10, 0., 10.);
   hRAL_pt_elecs_ = dbe_->book1D("RAL_pt_elecs", "RAL_pt_elecs", 50, 0., 300.);
   hRAL_eta_elecs_ = dbe_->book1D("RAL_eta_elecs", "RAL_eta_elecs", 50, -2.5, 2.5);
   hRAL_phi_elecs_ = dbe_->book1D("RAL_phi_elecs", "RAL_phi_elecs", 50, -4., 4.);
   hRAL_Iso_elecs_ = dbe_->book1D("RAL_Iso_elecs", "RAL_Iso_elecs", 50, 0., 25.);

   hRAL_Sum_pt_jets_ = dbe_->book1D("RAL_Sum_pt_jets", "RAL_Sum_pt_jets", 50, 0., 2000.);
   hRAL_Met_ = dbe_->book1D("RAL_Met", "RAL_Met", 50, 0., 1000.);

   hRAL_dR_emu_ = dbe_->book1D("RAL_deltaR_emu", "RAL_deltaR_emu", 50, 0., 10.);

   hRAL_mass_OS_mumu_ = dbe_->book1D("RAL_mass_OS_mumu", "RAL_mass_OS_mumu", 50, 0., 300.);
   hRAL_mass_OS_ee_ = dbe_->book1D("RAL_mass_OS_ee", "RAL_mass_OS_ee", 50, 0., 300.);
   hRAL_mass_OS_emu_ = dbe_->book1D("RAL_mass_OS_emu", "RAL_mass_OS_emu", 50, 0., 300.);
   hRAL_mass_SS_mumu_ = dbe_->book1D("RAL_mass_SS_mumu", "RAL_mass_SS_mumu", 50, 0., 300.);
   hRAL_mass_SS_ee_ = dbe_->book1D("RAL_mass_SS_ee", "RAL_mass_SS_ee", 50, 0., 300.);
   hRAL_mass_SS_emu_ = dbe_->book1D("RAL_mass_SS_emu", "RAL_mass_SS_emu", 50, 0., 300.);

   hRAL_Muon_monitor_ = dbe_->book2D("RAL_Single_Muon_Selection", "RAL_Single_Muon_Selection", 50, 0., 1000., 50, 0.,
         1000.);
   hRAL_Electron_monitor_ = dbe_->book2D("RAL_Single_Electron_Selection", "RAL_Single_Electron_Selection", 50, 0.,
         1000., 50, 0., 1000.);
   hRAL_OSee_monitor_ = dbe_->book2D("RAL_OS_Electron_Selection", "RAL_OS_Electron_Selection", 50, 0., 1000., 50, 0.,
         1000.);
   hRAL_OSemu_monitor_ = dbe_->book2D("RAL_OS_ElectronMuon_Selection", "RAL_OS_ElectronMuon_Selection", 50, 0., 1000.,
         50, 0., 1000.);
   hRAL_OSmumu_monitor_ = dbe_->book2D("RAL_OS_Muon_Selection", "RAL_OS_Muon_Selection", 50, 0., 1000., 50, 0., 1000.);
   hRAL_SSee_monitor_ = dbe_->book2D("RAL_SS_Electron_Selection", "RAL_SS_Electron_Selection", 50, 0., 1000., 50, 0.,
         1000.);
   hRAL_SSemu_monitor_ = dbe_->book2D("RAL_SS_ElectronMuon_Selection", "RAL_SS_ElectronMuon_Selection", 50, 0., 1000.,
         50, 0., 1000.);
   hRAL_SSmumu_monitor_ = dbe_->book2D("RAL_SS_Muon_Selection", "RAL_SS_Muon_Selection", 50, 0., 1000., 50, 0., 1000.);
   hRAL_TriMuon_monitor_ = dbe_->book2D("RAL_Tri_Muon_Selection", "RAL_Tri_Muon_Selection", 50, 0., 1000., 50, 0.,
         1000.);

}

template<typename Mu, typename Ele, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Jet, Met>::beginRun(const edm::Run& run) {

}

template<typename Mu, typename Ele, typename Jet, typename Met>
bool SusyDQM<Mu, Ele, Jet, Met>::goodSusyElectron(const Ele* ele) {
   //   if (ele->pt() < elec_pt_cut_)
   //      return false;
   if (fabs(ele->eta()) > elec_eta_cut_)
      return false;
   //   if (ele->mva() < elec_mva_cut_)
   //      return false;
   if (fabs(ele->gsfTrack()->dxy(bs)) > elec_d0_cut_)
      return false;
   return true;
}

template<typename Mu, typename Ele, typename Jet, typename Met>
bool SusyDQM<Mu, Ele, Jet, Met>::goodSusyMuon(const Mu* mu) {
   //   if (mu->pt() < muon_pt_cut_)
   //      return false;
   if (fabs(mu->eta()) > muon_eta_cut_)
      return false;
   if (!mu->isGlobalMuon())
      return false;
   if (mu->innerTrack()->numberOfValidHits() < muon_nHits_cut_)
      return false;
   if (mu->globalTrack()->normalizedChi2() > muon_nChi2_cut_)
      return false;
   if (fabs(mu->innerTrack()->dxy(bs)) > muon_d0_cut_)
      return false;
   return true;
}

template<typename Mu, typename Ele, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Jet, Met>::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {

   edm::Handle<std::vector<Mu> > muons;
   bool isFound = evt.getByLabel(muons_, muons);
   if (!isFound)
      return;

   edm::Handle<std::vector<Ele> > elecs;
   isFound = evt.getByLabel(electrons_, elecs);
   if (!isFound)
      return;

   //edm::Handle<std::vector<Jet> > jets;
   //evt.getByLabel(jets_, jets);

   //// sorted jets
   edm::Handle<std::vector<Jet> > cJets;
   isFound = evt.getByLabel(jets_, cJets);
   if (!isFound)
      return;
   std::vector<Jet> jets = *cJets;
   std::sort(jets.begin(), jets.end(), PtGreater());

   edm::Handle<std::vector<Met> > mets;
   isFound = evt.getByLabel(met_, mets);
   if (!isFound)
      return;

   edm::Handle<reco::VertexCollection> vertices;
   isFound = evt.getByLabel(vertex_, vertices);
   if (!isFound)
      return;

   //////////////////////////////
   // Hadronic DQM histos
   //////////////////////////////

   float RA1_HT = 0.;
   math::PtEtaPhiMLorentzVector RA1_vMHT(0., 0., 0., 0.);
   int RA1_nJets = 0;
   float RA1_jet1_pt = 0;
   float RA1_jet1_eta = 0;
   float RA1_jet1_emf = 0;
   float RA1_jet2_pt = 0;
   float RA1_jet2_eta = 0;
   float RA1_jet2_emf = 0;
   float RA1_jet3_pt = 0;
   math::PtEtaPhiMLorentzVector RA1_leading(0., 0., 0., 0.);
   math::PtEtaPhiMLorentzVector RA1_second(0., 0., 0., 0.);
   int i_jet = 0;
   for (typename std::vector<Jet>::const_iterator jet_i = jets.begin(); jet_i != jets.end(); ++jet_i) {
      if (i_jet == 0) {
         RA1_leading = jet_i->p4();
         RA1_jet1_pt = jet_i->pt();
         RA1_jet1_eta = jet_i->eta();
         RA1_jet1_emf = jet_i->emEnergyFraction();
      }
      if (i_jet == 1) {
         RA1_second = jet_i->p4();
         RA1_jet2_pt = jet_i->pt();
         RA1_jet2_eta = jet_i->eta();
         RA1_jet2_emf = jet_i->emEnergyFraction();
      }
      if (i_jet == 2)
         RA1_jet3_pt = jet_i->pt();
      if (jet_i->pt() > RA1_jet_pt_cut_ && fabs(jet_i->eta()) < RA1_jet_eta_cut_) {
         ++RA1_nJets;
         RA1_HT += jet_i->pt();
         RA1_vMHT -= jet_i->p4();
      }
      ++i_jet;
   }
   float RA1_MHT = RA1_vMHT.pt();

   i_jet = 0;
   float RA1_minDeltaPhi = 9999.;
   for (typename std::vector<Jet>::const_iterator jet_i = jets.begin(); jet_i != jets.end(); ++jet_i) {
      if (i_jet <= 2) {
         double deltaPhi_tmp = fabs(deltaPhi(jet_i->phi(), RA1_vMHT.phi()));
         if (deltaPhi_tmp < RA1_minDeltaPhi)
            RA1_minDeltaPhi = deltaPhi_tmp;
      }
      ++i_jet;
   }

   float RA1_alphat = 0;
   float RA1_DeltaPhiJets = 9999.;
   if (RA1_nJets >= 2) {
      RA1_DeltaPhiJets = fabs(deltaPhi(RA1_leading.phi(), RA1_second.phi()));
      // wrong definition PDG (July 2008) Eq. 38.38
      //RA1_alphat = RA1_second.Et() / (RA1_leading + RA1_second).Mt();
      // right definition PDG (July 2008) Eq. 38.61
      RA1_alphat = RA1_second.Et() / sqrt(2* RA1_leading .Et() * RA1_second.Et() * (1 - cos(RA1_DeltaPhiJets)));
   }

   float RA2_HT = 0.;
   math::PtEtaPhiMLorentzVector RA2_vMHT(0., 0., 0., 0.);
   int RA2_nJets = 0;
   float RA2_jet1_pt = 0;
   float RA2_jet1_eta = 0;
   float RA2_jet1_emf = 0;
   float RA2_jet2_pt = 0;
   float RA2_jet2_eta = 0;
   float RA2_jet2_emf = 0;
   float RA2_jet3_pt = 0;
   float RA2_jet3_eta = 0;
   float RA2_jet3_emf = 0;
   i_jet = 0;
   for (typename std::vector<Jet>::const_iterator jet_i = jets.begin(); jet_i != jets.end(); ++jet_i) {
      if (i_jet == 0) {
         RA2_jet1_pt = jet_i->pt();
         RA2_jet1_eta = jet_i->eta();
         RA2_jet1_emf = jet_i->emEnergyFraction();
      }
      if (i_jet == 1) {
         RA2_jet2_pt = jet_i->pt();
         RA2_jet2_eta = jet_i->eta();
         RA2_jet2_emf = jet_i->emEnergyFraction();
      }
      if (i_jet == 2) {
         RA2_jet3_pt = jet_i->pt();
         RA2_jet3_eta = jet_i->eta();
         RA2_jet3_emf = jet_i->emEnergyFraction();
      }
      if (jet_i->pt() > RA2_jet_pt_cut_ && fabs(jet_i->eta()) < RA2_jet_eta_cut_) {
         ++RA2_nJets;
         RA2_HT += jet_i->pt();
         RA2_vMHT -= jet_i->p4();
      }
      ++i_jet;
   }
   float RA2_MHT = RA2_vMHT.pt();

   i_jet = 0;
   float RA2_minDeltaPhi = 9999.;
   for (typename std::vector<Jet>::const_iterator jet_i = jets.begin(); jet_i != jets.end(); ++jet_i) {
      if (jet_i->pt() < RA2_jet_pt_cut_)
         continue;
      if (i_jet <= 2) {
         double deltaPhi_tmp = fabs(deltaPhi(jet_i->phi(), RA1_vMHT.phi()));
         if (deltaPhi_tmp < RA2_minDeltaPhi)
            RA2_minDeltaPhi = deltaPhi_tmp;
      }
      ++i_jet;
   }

   for (reco::VertexCollection::const_iterator vertex = vertices->begin(); vertex != vertices->end(); ++vertex) {
      bs = vertex->position();
      break;
   }

   float leadingMuPt = 0;
   for (typename std::vector<Mu>::const_iterator mu_i = muons->begin(); mu_i != muons->end(); ++mu_i) {
      if (!goodSusyMuon(&(*mu_i)))
         continue;

      reco::MuonIsolation Iso_muon = mu_i->isolationR03();
      float muIso = (Iso_muon.emEt + Iso_muon.hadEt + Iso_muon.sumPt) / mu_i->pt();

      if (muIso < RA12_muon_iso_cut_) {
         if (mu_i->pt() > leadingMuPt)
            leadingMuPt = mu_i->pt();
      }
   }

   float leadingElecPt = 0;
   for (typename std::vector<Ele>::const_iterator ele_i = elecs->begin(); ele_i != elecs->end(); ++ele_i) {
      if (!goodSusyElectron(&(*ele_i)))
         continue;

      float elecIso = (ele_i->dr03TkSumPt() + ele_i->dr03EcalRecHitSumEt() + ele_i->dr03HcalTowerSumEt()) / ele_i->pt();

      if (elecIso < RA12_elec_iso_cut_) {
         if (ele_i->pt() > leadingElecPt)
            leadingElecPt = ele_i->pt();
      }
   }

   //// Fill N-1 hsitograms for RA1
   if (RA1_jet1_emf >= RA1_jet_min_emf_cut_ && RA1_jet1_emf <= RA1_jet_max_emf_cut_ && RA1_jet2_emf
         >= RA1_jet_min_emf_cut_ && RA1_jet2_emf <= RA1_jet_max_emf_cut_) {
      hRA1_pt_jet1_->Fill(RA1_jet1_pt);
      hRA1_eta_jet1_->Fill(RA1_jet1_eta);
      hRA1_pt_jet2_->Fill(RA1_jet2_pt);
      hRA1_eta_jet2_->Fill(RA1_jet2_eta);
      hRA1_pt_jet3_->Fill(RA1_jet3_pt);
      hRA1_deltaPhi_mht_jets_->Fill(RA1_minDeltaPhi);
      hRA1_deltaPhi_jets_->Fill(RA1_DeltaPhiJets);
      hRA1_ht_->Fill(RA1_HT);
      hRA1_mht_->Fill(RA1_MHT);
      hRA1_alphat_->Fill(RA1_alphat);
      hRA1_pt_muons_->Fill(leadingMuPt);
      hRA1_pt_elecs_->Fill(leadingElecPt);
      for (int i = 0; i < 12; ++i) {
         if (RA1_jet1_pt > RA1_jet1_pt_cut_ || i == 0) {
            if (fabs(RA1_jet1_eta) < RA1_jet1_eta_cut_ || i == 1) {
               if (RA1_jet2_pt > RA1_jet2_pt_cut_ || i == 2) {
                  if (fabs(RA1_jet2_eta) < RA1_jet2_eta_cut_ || i == 3) {
                     if (RA1_jet3_pt < RA1_jet3_pt_cut_ || i == 4) {
                        if (RA1_minDeltaPhi >= RA1_deltaPhi_cut_ || i == 5) {
                           if (RA1_DeltaPhiJets <= RA1_deltaPhiJets_cut_ || i == 6) {
                              if (RA1_HT >= RA1_ht_cut_ || i == 7) {
                                 if (RA1_MHT >= RA1_mht_cut_ || i == 8) {
                                    if (RA1_alphat >= RA1_alphat_cut_ || i == 9) {
                                       if (leadingMuPt <= RA12_muon_pt_cut_ || i == 10) {
                                          if (leadingElecPt <= RA12_elec_pt_cut_ || i == 11) {
                                             if (i == 0)
                                                hRA1_pt_jet1_nm1_->Fill(RA1_jet1_pt);
                                             if (i == 1)
                                                hRA1_eta_jet1_nm1_->Fill(RA1_jet1_eta);
                                             if (i == 2)
                                                hRA1_pt_jet2_nm1_->Fill(RA1_jet2_pt);
                                             if (i == 3)
                                                hRA1_eta_jet2_nm1_->Fill(RA1_jet2_eta);
                                             if (i == 4)
                                                hRA1_pt_jet3_nm1_->Fill(RA1_jet3_pt);
                                             if (i == 5)
                                                hRA1_deltaPhi_mht_jets_nm1_->Fill(RA1_minDeltaPhi);
                                             if (i == 6)
                                                hRA1_deltaPhi_jets_nm1_->Fill(RA1_DeltaPhiJets);
                                             if (i == 7)
                                                hRA1_ht_nm1_->Fill(RA1_HT);
                                             if (i == 8)
                                                hRA1_mht_nm1_->Fill(RA1_MHT);
                                             if (i == 9)
                                                hRA1_alphat_nm1_->Fill(RA1_alphat);
                                             if (i == 10)
                                                hRA1_pt_muons_nm1_->Fill(leadingMuPt);
                                             if (i == 11)
                                                hRA1_pt_elecs_nm1_->Fill(leadingElecPt);
                                          }
                                       }
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //// Fill N-1 hsitograms for RA2
   if (RA2_jet1_emf >= RA2_jet_min_emf_cut_ && RA2_jet1_emf <= RA2_jet_max_emf_cut_ && RA2_jet2_emf
         >= RA2_jet_min_emf_cut_ && RA2_jet2_emf <= RA2_jet_max_emf_cut_ && RA2_jet3_emf >= RA2_jet_min_emf_cut_
         && RA2_jet3_emf <= RA2_jet_max_emf_cut_) {
      hRA2_N_jets_->Fill(RA2_nJets);
      hRA2_pt_jet1_->Fill(RA2_jet1_pt);
      hRA2_eta_jet1_->Fill(RA2_jet1_eta);
      hRA2_pt_jet2_->Fill(RA2_jet2_pt);
      hRA2_eta_jet2_->Fill(RA2_jet2_eta);
      hRA2_pt_jet3_->Fill(RA2_jet3_pt);
      hRA2_eta_jet3_->Fill(RA2_jet3_eta);
      hRA2_deltaPhi_mht_jets_->Fill(RA2_minDeltaPhi);
      hRA2_ht_->Fill(RA2_HT);
      hRA2_mht_->Fill(RA2_MHT);
      hRA2_pt_muons_->Fill(leadingMuPt);
      hRA2_pt_elecs_->Fill(leadingElecPt);
      for (int i = 0; i < 12; ++i) {
         if (RA2_nJets >= RA2_N_jets_cut_ || i == 0) {
            if (RA2_jet1_pt > RA2_jet1_pt_cut_ || i == 1) {
               if (fabs(RA2_jet1_eta) < RA2_jet1_eta_cut_ || i == 2) {
                  if (RA2_jet2_pt > RA2_jet2_pt_cut_ || i == 3) {
                     if (fabs(RA2_jet2_eta) < RA2_jet2_eta_cut_ || i == 4) {
                        if (RA2_jet3_pt > RA2_jet3_pt_cut_ || i == 5) {
                           if (fabs(RA2_jet3_eta) < RA2_jet3_eta_cut_ || i == 6) {
                              if (RA2_minDeltaPhi >= RA2_deltaPhi_cut_ || i == 7) {
                                 if (RA2_HT >= RA2_ht_cut_ || i == 8) {
                                    if (RA2_MHT >= RA2_mht_cut_ || i == 9) {
                                       if (leadingMuPt <= RA12_muon_pt_cut_ || i == 10) {
                                          if (leadingElecPt <= RA12_elec_pt_cut_ || i == 11) {
                                             if (i == 0)
                                                hRA2_N_jets_nm1_->Fill(RA2_nJets);
                                             if (i == 1)
                                                hRA2_pt_jet1_nm1_->Fill(RA2_jet1_pt);
                                             if (i == 2)
                                                hRA2_eta_jet1_nm1_->Fill(RA2_jet1_eta);
                                             if (i == 3)
                                                hRA2_pt_jet2_nm1_->Fill(RA2_jet2_pt);
                                             if (i == 4)
                                                hRA2_eta_jet2_nm1_->Fill(RA2_jet2_eta);
                                             if (i == 5)
                                                hRA2_pt_jet3_nm1_->Fill(RA2_jet3_pt);
                                             if (i == 6)
                                                hRA2_eta_jet3_nm1_->Fill(RA2_jet3_eta);
                                             if (i == 7)
                                                hRA2_deltaPhi_mht_jets_nm1_->Fill(RA2_minDeltaPhi);
                                             if (i == 8)
                                                hRA2_ht_nm1_->Fill(RA2_HT);
                                             if (i == 9)
                                                hRA2_mht_nm1_->Fill(RA2_MHT);
                                             if (i == 10)
                                                hRA2_pt_muons_nm1_->Fill(leadingMuPt);
                                             if (i == 11)
                                                hRA2_pt_elecs_nm1_->Fill(leadingElecPt);
                                          }
                                       }
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   //////////////////////////////
   // Leptonic DQM histos
   //////////////////////////////

   float sumPt = 0.;
   for (typename std::vector<Jet>::const_iterator jet_i = jets.begin(); jet_i != jets.end(); ++jet_i) {
      if (jet_i->pt() < RAL_jet_pt_cut_)
         continue;
      if (fabs(jet_i->eta()) > RAL_jet_eta_cut_)
         continue;
      if (fabs(jet_i->eta()) > RAL_jet_eta_cut_)
         continue;
      if (jet_i->emEnergyFraction() < RAL_jet_min_emf_cut_)
         continue;
      if (jet_i->emEnergyFraction() > RAL_jet_max_emf_cut_)
         continue;
      sumPt += jet_i->pt();
   }

   hRAL_Sum_pt_jets_->Fill(sumPt);

   float MET = 0.;
   for (typename std::vector<Met>::const_iterator met_i = mets->begin(); met_i != mets->end(); ++met_i) {
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

   for (typename std::vector<Mu>::const_iterator mu_i = muons->begin(); mu_i != muons->end(); ++mu_i) {
      if (!(goodSusyMuon(&(*mu_i)) && mu_i->pt() > RAL_muon_pt_cut_))
         continue;
      ++nMuons;

      hRAL_pt_muons_->Fill(mu_i->pt());
      hRAL_eta_muons_->Fill(mu_i->eta());
      hRAL_phi_muons_->Fill(mu_i->phi());

      reco::MuonIsolation muIso = mu_i->isolationR03();
      hRAL_Iso_muons_->Fill(muIso.emEt + muIso.hadEt + muIso.sumPt);

      //Muon muon pairs
      for (typename std::vector<Mu>::const_iterator mu_j = muons->begin(); mu_j != muons->end(); ++mu_j) {
         if (mu_i >= mu_j)
            continue;
         if (!(goodSusyMuon(&(*mu_j)) && mu_j->pt() > RAL_muon_pt_cut_))
            continue;

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

      //Electron muon pairs
      for (typename std::vector<Ele>::const_iterator ele_j = elecs->begin(); ele_j != elecs->end(); ++ele_j) {
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
   for (typename std::vector<Ele>::const_iterator ele_i = elecs->begin(); ele_i != elecs->end(); ++ele_i) {
      if (!(goodSusyElectron(&(*ele_i)) && ele_i->pt() > RAL_elec_pt_cut_))
         continue;
      nElectrons++;

      hRAL_pt_elecs_->Fill(ele_i->pt());
      hRAL_eta_elecs_->Fill(ele_i->eta());
      hRAL_phi_elecs_->Fill(ele_i->phi());

      hRAL_Iso_elecs_->Fill(ele_i->dr03TkSumPt() + ele_i->dr03EcalRecHitSumEt() + ele_i->dr03HcalTowerSumEt());

      //Electron electron pairs
      for (typename std::vector<Ele>::const_iterator ele_j = elecs->begin(); ele_j != elecs->end(); ++ele_j) {
         if (ele_i >= ele_j)
            continue;
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

template<typename Mu, typename Ele, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Jet, Met>::endRun(const edm::Run& run) {

}

template<typename Mu, typename Ele, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Jet, Met>::endJob() {

}

#endif

typedef SusyDQM<reco::Muon, reco::GsfElectron, reco::CaloJet, reco::CaloMET> RecoSusyDQM;
//typedef SusyDQM< pat::Muon, pat::Electron, pat::Jet, pat::MET > PatSusyDQM;
