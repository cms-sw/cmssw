#ifndef DQM_Physics_NPSDQM_h
#define DQM_Physics_NPSDQM_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <string>
#include <vector>

class NPSDQM : public DQMEDAnalyzer {
public:
  explicit NPSDQM(const edm::ParameterSet&);
  ~NPSDQM() override = default;

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  std::vector<std::string> jetLabels_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Jet>>> jetTokens_;
  edm::EDGetTokenT<reco::JetTagCollection> btagToken_;
  edm::EDGetTokenT<std::vector<reco::PFMET>> PFMETToken_;
  edm::EDGetTokenT<edm::View<reco::Muon>> muonToken_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron>> electronToken_;
  edm::EDGetTokenT<edm::View<reco::Photon>> photonToken_;

  std::vector<MonitorElement*> Jet_pt, Jet_eta, Jet_phi, Jet_m;
  std::vector<MonitorElement*> Jet_chef, Jet_nhef, Jet_cemf, Jet_nemf;
  
  // Jet 1 Kinematics
  std::vector<MonitorElement*> Jet1_pt, Jet1_eta, Jet1_phi;
  
  // dPhi(Jet, MET)
  std::vector<MonitorElement*> dPhi_Jet1_MET, dPhi_Jet2_MET, dPhi_Jet3_MET, dPhi_Jet4_MET;
  
  std::vector<MonitorElement*> CentralJet_HT;
  std::vector<MonitorElement*> ForwardJet_HT;
  std::vector<MonitorElement*> N_CentralJets;
  std::vector<MonitorElement*> N_ForwardJets;
  std::vector<MonitorElement*> N_bjets; 

  MonitorElement* MET_pt;
  MonitorElement* MET_phi;

  MonitorElement* pt_muons[3];
  MonitorElement* eta_muons[3];
  MonitorElement* phi_muons[3];
  MonitorElement* mt_muons[3];

  MonitorElement* pt_elecs[3];
  MonitorElement* eta_elecs[3];
  MonitorElement* phi_elecs[3];
  MonitorElement* mt_elecs[3];

  MonitorElement* LT;
  MonitorElement* N_Leptons;

  MonitorElement* pt_photons;
  MonitorElement* eta_photons;
  MonitorElement* phi_photons;
  MonitorElement* N_photons;
};

#endif
