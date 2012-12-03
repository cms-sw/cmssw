#ifndef ExoticaDQM_H
#define ExoticaDQM_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

// Trigger stuff
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Common/interface/Handle.h" 
#include "FWCore/Framework/interface/DataKeyTags.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include <DataFormats/EgammaCandidates/interface/GsfElectron.h>

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DQMServices/Core/interface/MonitorElement.h"

// ParticleFlow
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

// EGamma
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

// Muon
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h" 

// Tau
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"

// Jets
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "RecoJets/JetProducers/interface/JetIDHelper.h"

// Photon
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

// MET
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"

//
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DQMStore;
 
class ExoticaDQM: public edm::EDAnalyzer{

public:

  ExoticaDQM(const edm::ParameterSet& ps);
  virtual ~ExoticaDQM();
  
protected:

  virtual void beginJob();
  virtual void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;
  virtual void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  virtual void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
  virtual void endJob();
  
  //Diagnostic
  virtual void analyzeMultiJets(edm::Event const& e);
  virtual void analyzeMultiJetsTrigger(edm::Event const& e);
  
  virtual void analyzeLongLived(edm::Event const& e);
  virtual void analyzeLongLivedTrigger(edm::Event const& e); 

  virtual void analyzeEventInterpretation(edm::Event const& e, edm::EventSetup const& eSetup);
  
  //
  //virtual void analyzeTopLike(edm::Event const& e);
  //virtual void analyzeTopLikeTrigger(edm::Event const& e);
  //
  //virtual void analyzeLeptonJet(edm::Event const& e);
  //virtual void analyzeLeptonJetTrigger(edm::Event const& e);
  //
  //virtual void analyzeNonHadronic(edm::Event const& e);
  //virtual void analyzeNonHadronicTrigger(edm::Event const& e);

private:

  void bookHistos(DQMStore * bei );
  
  unsigned long long m_cacheID_;
  int nLumiSecs_;
  int nEvents_, irun, ievt;
  reco::CandidateCollection *leptonscands_;
  int leptonflavor;
  float pi;
  
  DQMStore* bei_;  
  HLTConfigProvider hltConfigProvider_;
  bool isValidHltConfig_;

  // Variables from config file
  edm::InputTag theTriggerResultsCollection;
  std::vector<std::string>  theTriggerForMultiJetsList;
  std::vector<std::string>  theTriggerForLongLivedList;
  edm::Handle<edm::TriggerResults> triggerResults_;


  // Electrons
  edm::InputTag ElectronLabel_;
  edm::Handle<reco::GsfElectronCollection> ElectronCollection_;
  //
  edm::InputTag PFElectronLabelEI_;
  edm::Handle<reco::PFCandidateCollection> pfElectronCollectionEI_;
  reco::PFCandidateCollection pfelectronsEI; 
  
  
  // Muons
  edm::InputTag MuonLabel_;
  edm::Handle<reco::MuonCollection> MuonCollection_;
  //
  edm::InputTag PFMuonLabelEI_;
  edm::Handle<reco::PFCandidateCollection> pfMuonCollectionEI_;
  reco::PFCandidateCollection pfmuonsEI; 

  
  // Taus
  edm::InputTag TauLabel_;
  edm::Handle<reco::CaloTauCollection> TauCollection_;
  //
  edm::InputTag PFTauLabelEI_;
  edm::Handle<reco::PFTauCollection> pfTauCollectionEI_;

  
  // Photons
  edm::InputTag PhotonLabel_;
  edm::Handle<reco::PhotonCollection> PhotonCollection_;
  //
  edm::InputTag PFPhotonLabelEI_;
  edm::Handle<reco::PFCandidateCollection> pfPhotonCollectionEI_;
  reco::PFCandidateCollection pfphotons;

  
  // Jets
  edm::InputTag CaloJetLabel_;
  edm::Handle<reco::CaloJetCollection> caloJetCollection_;
  reco::CaloJetCollection calojets; 
  //
  edm::InputTag PFJetLabel_; 
  edm::Handle<reco::PFJetCollection> pfJetCollection_;
  reco::PFJetCollection pfjets;
  //
  edm::InputTag PFJetLabelEI_; 
  edm::Handle<reco::PFJetCollection> pfJetCollectionEI_;
  reco::PFJetCollection pfjetsEI;
  
  
  // MET
  edm::InputTag CaloMETLabel_;
  edm::Handle<reco::CaloMETCollection> caloMETCollection_;
  //
  edm::InputTag PFMETLabel_;
  edm::Handle<reco::PFMETCollection> pfMETCollection_; 
  //
  edm::InputTag PFMETLabelEI_;
  edm::Handle<reco::PFMETCollection> pfMETCollectionEI_;
  
  ///////////////////////////
  // Parameters 
  ///////////////////////////
  // Cuts - MultiJets
  // inputs
  std::string CaloJetCorService_;
  std::string PFJetCorService_;
  reco::helper::JetIDHelper *jetID;
  double mj_monojet_ptPFJet_;
  double mj_monojet_ptPFMuon_;
  double mj_monojet_ptPFElectron_;
  //
  int    mj_monojet_countPFJet;
  //
  double CaloJetPx[2];
  double CaloJetPy[2];
  double CaloJetPt[2];
  double CaloJetEta[2];
  double CaloJetPhi[2];
  double CaloJetEMF[2];
  double CaloJetfHPD[2];
  double CaloJetn90[2];
  //
  double PFJetPx[2];
  double PFJetPy[2];
  double PFJetPt[2];
  double PFJetEta[2];
  double PFJetPhi[2];
  double PFJetNHEF[2];
  double PFJetCHEF[2];
  double PFJetNEMF[2];
  double PFJetCEMF[2];

  // Cuts - Long Lived
  //

  // Cuts - EI
  //
  double PFJetEIPx;
  double PFJetEIPy;
  double PFJetEIPt;
  double PFJetEIEta;
  double PFJetEIPhi;
  double PFJetEINHEF;
  double PFJetEICHEF;
  double PFJetEINEMF;
  double PFJetEICEMF;
  
  ///////////////////////////
  // Histograms
  ///////////////////////////
  // Histograms - MultiJets
  //
  MonitorElement* mj_monojet_pfchef;
  MonitorElement* mj_monojet_pfnhef;
  MonitorElement* mj_monojet_pfcemf;
  MonitorElement* mj_monojet_pfnemf;
  MonitorElement* mj_monojet_pfJet1_pt;
  MonitorElement* mj_monojet_pfJet2_pt;
  MonitorElement* mj_monojet_pfJet1_eta;
  MonitorElement* mj_monojet_pfJet2_eta;
  MonitorElement* mj_monojet_pfJetMulti;
  MonitorElement* mj_monojet_deltaPhiPFJet1PFJet2;
  MonitorElement* mj_monojet_deltaRPFJet1PFJet2;
  MonitorElement* mj_monojet_pfmetnomu;
  MonitorElement* mj_caloMet_et;
  MonitorElement* mj_caloMet_phi;
  MonitorElement* mj_pfMet_et;
  MonitorElement* mj_pfMet_phi;
  // Histograms - MultiJets Trigger
  //
  // Histograms - LongLived
  //
  MonitorElement* ll_gammajet_sMajMajPhot; 
  MonitorElement* ll_gammajet_sMinMinPhot;  
  // Histograms - LongLived Trigger
  //
  // Histograms - EIComparison
  MonitorElement* ei_pfjet1_pt;
  MonitorElement* ei_pfmet_pt;
  MonitorElement* ei_pfmuon_pt;
  MonitorElement* ei_pfelectron_pt;
  
};


#endif
