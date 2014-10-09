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

  //Resonances
  virtual void analyzeDiJets(edm::Event const& e);
  virtual void analyzeDiMuons(edm::Event const& e);
  virtual void analyzeDiElectrons(edm::Event const& e);

  //Mono Searches
  virtual void analyzeMonoJets(edm::Event const& e);
  virtual void analyzeMonoMuons(edm::Event const& e);
  virtual void analyzeMonoElectrons(edm::Event const& e);

  //Other... Phat stuff.
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
  edm::EDGetTokenT<reco::GsfElectronCollection> ElectronToken_;
  edm::Handle<reco::GsfElectronCollection> ElectronCollection_;
  //
  edm::EDGetTokenT<reco::PFCandidateCollection> PFElectronTokenEI_;
  edm::Handle<reco::PFCandidateCollection> pfElectronCollectionEI_;
  reco::PFCandidateCollection pfelectronsEI;


  // Muons
  edm::EDGetTokenT<reco::MuonCollection> MuonToken_;
  edm::Handle<reco::MuonCollection> MuonCollection_;
  //
  edm::EDGetTokenT<reco::PFCandidateCollection> PFMuonTokenEI_;
  edm::Handle<reco::PFCandidateCollection> pfMuonCollectionEI_;
  reco::PFCandidateCollection pfmuonsEI;


  // Taus
  edm::EDGetTokenT<reco::CaloTauCollection> TauToken_;
  edm::Handle<reco::CaloTauCollection> TauCollection_;
  //
  edm::InputTag PFTauLabelEI_;
  edm::Handle<reco::PFTauCollection> pfTauCollectionEI_;


  // Photons
  edm::EDGetTokenT<reco::PhotonCollection> PhotonToken_;
  edm::Handle<reco::PhotonCollection> PhotonCollection_;
  //
  edm::InputTag PFPhotonLabelEI_;
  edm::Handle<reco::PFCandidateCollection> pfPhotonCollectionEI_;
  reco::PFCandidateCollection pfphotons;


  // Jets
  edm::EDGetTokenT<reco::CaloJetCollection> CaloJetToken_;
  edm::Handle<reco::CaloJetCollection> caloJetCollection_;
  reco::CaloJetCollection calojets;
  //
  edm::EDGetTokenT<reco::PFJetCollection> PFJetToken_;
  edm::Handle<reco::PFJetCollection> pfJetCollection_;
  reco::PFJetCollection pfjets;
  //
  edm::EDGetTokenT<reco::PFJetCollection> PFJetTokenEI_;
  edm::Handle<reco::PFJetCollection> pfJetCollectionEI_;
  reco::PFJetCollection pfjetsEI;


  // MET
  edm::EDGetTokenT<reco::CaloMETCollection> CaloMETToken_;
  edm::Handle<reco::CaloMETCollection> caloMETCollection_;
  //
  edm::EDGetTokenT<reco::PFMETCollection> PFMETToken_;
  edm::Handle<reco::PFMETCollection> pfMETCollection_;
  //
  edm::EDGetTokenT<reco::PFMETCollection> PFMETTokenEI_;
  edm::Handle<reco::PFMETCollection> pfMETCollectionEI_;

  // ECAL RECHITS
  edm::EDGetTokenT<EBRecHitCollection> ecalBarrelRecHitToken_; // reducedEcalRecHitsEB
  edm::EDGetTokenT<EERecHitCollection> ecalEndcapRecHitToken_; // reducedEcalRecHitsEE

  ///////////////////////////
  // Parameters
  ///////////////////////////
  // Cuts - MultiJets
  // inputs
  std::string CaloJetCorService_;
  std::string PFJetCorService_;
  reco::helper::JetIDHelper *jetID;

  //Varibles Used
  // PFJets
  double PFJetPx[2];
  double PFJetPy[2];
  double PFJetPt[2];
  double PFJetEta[2];
  double PFJetPhi[2];
  double PFJetNHEF[2];
  double PFJetCHEF[2];
  double PFJetNEMF[2];
  double PFJetCEMF[2];

  // Muons
  //
  double MuonPx[2];
  double MuonPy[2];
  double MuonPt[2];
  double MuonEta[2];
  double MuonPhi[2];
  double MuonCharge[2];

  // Electrons
  //
  double ElectronPx[2];
  double ElectronPy[2];
  double ElectronPt[2];
  double ElectronEta[2];
  double ElectronPhi[2];
  double ElectronCharge[2];

  ///////////////////////////
  // Histograms
  ///////////////////////////
  // Histograms - Dijet
  //
  MonitorElement* dijet_PFJet1_pt;
  MonitorElement* dijet_PFJet1_eta;
  MonitorElement* dijet_PFJet1_phi;
  MonitorElement* dijet_PFJet2_pt;
  MonitorElement* dijet_PFJet2_eta;
  MonitorElement* dijet_PFJet2_phi;
  MonitorElement* dijet_deltaPhiPFJet1PFJet2;
  MonitorElement* dijet_deltaEtaPFJet1PFJet2;
  MonitorElement* dijet_deltaRPFJet1PFJet2;
  MonitorElement* dijet_invMassPFJet1PFJet2;
  MonitorElement* dijet_PFchef;
  MonitorElement* dijet_PFnhef;
  MonitorElement* dijet_PFcemf;
  MonitorElement* dijet_PFnemf;
  MonitorElement* dijet_PFJetMulti;
  //
  double dijet_PFJet1_pt_cut_;
  double dijet_PFJet2_pt_cut_;
  int    dijet_countPFJet_;

  ///////////////////////////
  // Histograms - DiMuon
  //
  MonitorElement* dimuon_Muon1_pt;
  MonitorElement* dimuon_Muon1_eta;
  MonitorElement* dimuon_Muon1_phi;
  MonitorElement* dimuon_Muon2_pt;
  MonitorElement* dimuon_Muon2_eta;
  MonitorElement* dimuon_Muon2_phi;
  MonitorElement* dimuon_Charge;
  MonitorElement* dimuon_deltaEtaMuon1Muon2;
  MonitorElement* dimuon_deltaPhiMuon1Muon2;
  MonitorElement* dimuon_deltaRMuon1Muon2; 
  MonitorElement* dimuon_invMassMuon1Muon2;
  MonitorElement* dimuon_MuonMulti;
  //
  double dimuon_Muon1_pt_cut_;
  double dimuon_Muon2_pt_cut_;
  int    dimuon_countMuon_;

  ///////////////////////////
  // Histograms - DiElectron
  //
  MonitorElement* dielectron_Electron1_pt;
  MonitorElement* dielectron_Electron1_eta;
  MonitorElement* dielectron_Electron1_phi;
  MonitorElement* dielectron_Electron2_pt;
  MonitorElement* dielectron_Electron2_eta;
  MonitorElement* dielectron_Electron2_phi;
  MonitorElement* dielectron_Charge;
  MonitorElement* dielectron_deltaEtaElectron1Electron2;
  MonitorElement* dielectron_deltaPhiElectron1Electron2;
  MonitorElement* dielectron_deltaRElectron1Electron2; 
  MonitorElement* dielectron_invMassElectron1Electron2;
  MonitorElement* dielectron_ElectronMulti;
  //
  double dielectron_Electron1_pt_cut_;
  double dielectron_Electron2_pt_cut_;
  int    dielectron_countElectron_;

  ///////////////////////////
  // Histograms - MonoJet
  //
  MonitorElement* monojet_PFJet_pt;
  MonitorElement* monojet_PFJet_eta;
  MonitorElement* monojet_PFJet_phi;
  MonitorElement* monojet_PFMet;
  MonitorElement* monojet_PFMet_phi;
  MonitorElement* monojet_PFJetPtOverPFMet;
  MonitorElement* monojet_deltaPhiPFJetPFMet;
  MonitorElement* monojet_PFchef;
  MonitorElement* monojet_PFnhef;
  MonitorElement* monojet_PFcemf;
  MonitorElement* monojet_PFnemf;
  MonitorElement* monojet_PFJetMulti;
  //
  double monojet_PFJet_pt_cut_;
  double monojet_PFJet_met_cut_;
  int    monojet_countPFJet_;

  ///////////////////////////
  // Histograms - MonoMuon
  //
  MonitorElement* monomuon_Muon_pt;
  MonitorElement* monomuon_Muon_eta;
  MonitorElement* monomuon_Muon_phi;
  MonitorElement* monomuon_Charge;
  MonitorElement* monomuon_PFMet;
  MonitorElement* monomuon_PFMet_phi;
  MonitorElement* monomuon_MuonPtOverPFMet;
  MonitorElement* monomuon_deltaPhiMuonPFMet;
  MonitorElement* monomuon_TransverseMass;
  MonitorElement* monomuon_MuonMulti;
  //
  double monomuon_Muon_pt_cut_;
  double monomuon_Muon_met_cut_;
  int    monomuon_countMuon_;

  /////////////////////////////
  // Histograms - MonoElectron
  //
  MonitorElement* monoelectron_Electron_pt;
  MonitorElement* monoelectron_Electron_eta;
  MonitorElement* monoelectron_Electron_phi;
  MonitorElement* monoelectron_Charge;
  MonitorElement* monoelectron_PFMet;
  MonitorElement* monoelectron_ElectronPtOverPFMet;
  MonitorElement* monoelectron_PFMet_phi;
  MonitorElement* monoelectron_deltaPhiElectronPFMet;
  MonitorElement* monoelectron_TransverseMass;
  MonitorElement* monoelectron_ElectronMulti;
  //
  double monoelectron_Electron_pt_cut_;
  double monoelectron_Electron_met_cut_;
  int    monoelectron_countElectron_;



  /////////// Phat Stuff /////////
  // Cuts - Long Lived
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

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
