#ifndef ExoticaDQM_H
#define ExoticaDQM_H

#include <memory>

// DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/DataKeyTags.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/Common/interface/TriggerNames.h"

// Candidate handling
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "RecoJets/JetProducers/interface/JetIDHelper.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
// Other
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

// ROOT
#include "TLorentzVector.h"

// STDLIB
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
#include <map>

class ExoticaDQM : public DQMEDAnalyzer {
public:
  ExoticaDQM(const edm::ParameterSet& ps);
  ~ExoticaDQM() override;

protected:
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;

  //Resonances
  virtual void analyzeDiJets(edm::Event const& e);
  virtual void analyzeDiMuons(edm::Event const& e);
  virtual void analyzeDiElectrons(edm::Event const& e);
  virtual void analyzeDiPhotons(edm::Event const& e);

  //Mono Searches
  virtual void analyzeMonoJets(edm::Event const& e);
  virtual void analyzeMonoMuons(edm::Event const& e);
  virtual void analyzeMonoElectrons(edm::Event const& e);
  virtual void analyzeMonoPhotons(edm::Event const& e);

  // Displaced Fermion Searches
  virtual void analyzeDisplacedLeptons(edm::Event const& e, const edm::EventSetup& s);
  virtual void analyzeDisplacedJets(edm::Event const& e, const edm::EventSetup& s);

  // Estimate the momentum vector that a GenParticle would have at its trajectory's point of closest approach to the beam-line.
  virtual GlobalVector getGenParticleTrajectoryAtBeamline(const edm::EventSetup& iSetup, const reco::GenParticle* gen);

private:
  void bookHistograms(DQMStore::IBooker& bei, edm::Run const&, edm::EventSetup const&) override;

  int nLumiSecs_;
  int nEvents_, irun, ievt;

  bool isValidHltConfig_;

  //Trigger
  std::vector<std::string> HltPaths_;
  edm::EDGetTokenT<edm::TriggerResults> TriggerToken_;
  edm::Handle<edm::TriggerResults> TriggerResults_;

  //Vertex
  edm::EDGetTokenT<reco::VertexCollection> VertexToken_;
  edm::Handle<reco::VertexCollection> VertexCollection_;

  // Electrons
  edm::EDGetTokenT<reco::GsfElectronCollection> ElectronToken_;
  edm::Handle<reco::GsfElectronCollection> ElectronCollection_;

  // Muons
  edm::EDGetTokenT<reco::MuonCollection> MuonToken_;
  edm::Handle<reco::MuonCollection> MuonCollection_;

  // Photons
  edm::EDGetTokenT<reco::PhotonCollection> PhotonToken_;
  edm::Handle<reco::PhotonCollection> PhotonCollection_;

  // Jets
  edm::EDGetTokenT<reco::CaloJetCollection> CaloJetToken_;
  edm::Handle<reco::CaloJetCollection> caloJetCollection_;
  reco::CaloJetCollection calojets;
  // Nominal Jets
  edm::EDGetTokenT<reco::PFJetCollection> PFJetToken_;
  edm::Handle<reco::PFJetCollection> pfJetCollection_;
  reco::PFJetCollection pfjets;

  //All Other Jets
  std::vector<edm::EDGetTokenT<reco::PFJetCollection> > DiJetPFJetToken_;
  std::vector<edm::InputTag> DiJetPFJetCollection_;
  edm::Handle<reco::PFJetCollection> DiJetpfJetCollection_;
  reco::PFJetCollection DiJetpfjets;

  // MET
  edm::EDGetTokenT<reco::CaloMETCollection> CaloMETToken_;
  edm::Handle<reco::CaloMETCollection> caloMETCollection_;
  //
  edm::EDGetTokenT<reco::PFMETCollection> PFMETToken_;
  edm::Handle<reco::PFMETCollection> pfMETCollection_;

  // ECAL RECHITS
  edm::EDGetTokenT<EBRecHitCollection> ecalBarrelRecHitToken_;  // reducedEcalRecHitsEB
  edm::EDGetTokenT<EERecHitCollection> ecalEndcapRecHitToken_;  // reducedEcalRecHitsEE

  edm::EDGetTokenT<reco::JetCorrector> JetCorrectorToken_;
  edm::Handle<reco::JetCorrector> JetCorrector_;

  // Tracks
  edm::EDGetTokenT<reco::TrackCollection> TrackToken_;
  edm::Handle<reco::TrackCollection> TrackCollection_;

  // Special collections for highly displaced particles
  edm::EDGetTokenT<reco::TrackCollection> MuonDispToken_;
  edm::Handle<reco::TrackCollection> MuonDispCollection_;
  edm::EDGetTokenT<reco::TrackCollection> MuonDispSAToken_;
  edm::Handle<reco::TrackCollection> MuonDispSACollection_;

  // MC truth
  edm::EDGetTokenT<reco::GenParticleCollection> GenParticleToken_;
  edm::Handle<reco::GenParticleCollection> GenCollection_;

  ///////////////////////////
  // Parameters
  ///////////////////////////
  // Cuts - MultiJets
  // inputs

  reco::helper::JetIDHelper* jetID;

  //Varibles Used
  // PFJets
  double PFJetPx[2];
  double PFJetPy[2];
  double PFJetPt[2];
  double PFJetEta[2];
  double PFJetPhi[2];
  double PFJetRapidity[2];
  double PFJetMass[2];
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

  // Photon
  //
  double PhotonEnergy[2];
  double PhotonPt[2];
  double PhotonEt[2];
  double PhotonEta[2];
  double PhotonEtaSc[2];
  double PhotonPhi[2];
  double PhotonHoverE[2];
  double PhotonSigmaIetaIeta[2];
  double PhotonTrkSumPtSolidConeDR03[2];
  double PhotonE1x5E5x5[2];
  double PhotonE2x5E5x5[2];

  ///////////////////////////
  // Histograms
  ///////////////////////////
  // Histograms - Dijet
  std::vector<MonitorElement*> dijet_PFJet_pt;
  std::vector<MonitorElement*> dijet_PFJet_eta;
  std::vector<MonitorElement*> dijet_PFJet_phi;
  std::vector<MonitorElement*> dijet_PFJet_rapidity;
  std::vector<MonitorElement*> dijet_PFJet_mass;
  std::vector<MonitorElement*> dijet_deltaPhiPFJet1PFJet2;
  std::vector<MonitorElement*> dijet_deltaEtaPFJet1PFJet2;
  std::vector<MonitorElement*> dijet_deltaRPFJet1PFJet2;
  std::vector<MonitorElement*> dijet_invMassPFJet1PFJet2;
  std::vector<MonitorElement*> dijet_PFchef;
  std::vector<MonitorElement*> dijet_PFnhef;
  std::vector<MonitorElement*> dijet_PFcemf;
  std::vector<MonitorElement*> dijet_PFnemf;
  std::vector<MonitorElement*> dijet_PFJetMulti;
  //
  double dijet_PFJet1_pt_cut_;
  double dijet_PFJet2_pt_cut_;
  int dijet_countPFJet_;

  ///////////////////////////
  // Histograms - DiMuon
  //
  MonitorElement* dimuon_Muon_pt;
  MonitorElement* dimuon_Muon_eta;
  MonitorElement* dimuon_Muon_phi;
  MonitorElement* dimuon_Charge;
  MonitorElement* dimuon_deltaEtaMuon1Muon2;
  MonitorElement* dimuon_deltaPhiMuon1Muon2;
  MonitorElement* dimuon_deltaRMuon1Muon2;
  MonitorElement* dimuon_invMassMuon1Muon2;
  MonitorElement* dimuon_MuonMulti;
  //
  double dimuon_Muon1_pt_cut_;
  double dimuon_Muon2_pt_cut_;
  int dimuon_countMuon_;

  ///////////////////////////
  // Histograms - DiElectron
  //
  MonitorElement* dielectron_Electron_pt;
  MonitorElement* dielectron_Electron_eta;
  MonitorElement* dielectron_Electron_phi;
  MonitorElement* dielectron_Charge;
  MonitorElement* dielectron_deltaEtaElectron1Electron2;
  MonitorElement* dielectron_deltaPhiElectron1Electron2;
  MonitorElement* dielectron_deltaRElectron1Electron2;
  MonitorElement* dielectron_invMassElectron1Electron2;
  MonitorElement* dielectron_ElectronMulti;
  //
  double dielectron_Electron1_pt_cut_;
  double dielectron_Electron2_pt_cut_;
  int dielectron_countElectron_;

  ///////////////////////////
  // Histograms - DiPhoton
  //
  MonitorElement* diphoton_Photon_pt;
  MonitorElement* diphoton_Photon_energy;
  MonitorElement* diphoton_Photon_et;
  MonitorElement* diphoton_Photon_eta;
  MonitorElement* diphoton_Photon_etasc;
  MonitorElement* diphoton_Photon_phi;
  MonitorElement* diphoton_Photon_hovere_eb;
  MonitorElement* diphoton_Photon_hovere_ee;
  MonitorElement* diphoton_Photon_sigmaietaieta_eb;
  MonitorElement* diphoton_Photon_sigmaietaieta_ee;
  MonitorElement* diphoton_Photon_trksumptsolidconedr03_eb;
  MonitorElement* diphoton_Photon_trksumptsolidconedr03_ee;
  MonitorElement* diphoton_Photon_e1x5e5x5_eb;
  MonitorElement* diphoton_Photon_e1x5e5x5_ee;
  MonitorElement* diphoton_Photon_e2x5e5x5_eb;
  MonitorElement* diphoton_Photon_e2x5e5x5_ee;
  MonitorElement* diphoton_deltaEtaPhoton1Photon2;
  MonitorElement* diphoton_deltaPhiPhoton1Photon2;
  MonitorElement* diphoton_deltaRPhoton1Photon2;
  MonitorElement* diphoton_invMassPhoton1Photon2;
  MonitorElement* diphoton_PhotonMulti;
  //
  double diphoton_Photon1_pt_cut_;
  double diphoton_Photon2_pt_cut_;
  int diphoton_countPhoton_;

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
  int monojet_countPFJet_;

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
  int monomuon_countMuon_;

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
  int monoelectron_countElectron_;

  ///////////////////////////
  // Histograms - DiPhoton
  //
  MonitorElement* monophoton_Photon_pt;
  MonitorElement* monophoton_Photon_energy;
  MonitorElement* monophoton_Photon_et;
  MonitorElement* monophoton_Photon_eta;
  MonitorElement* monophoton_Photon_etasc;
  MonitorElement* monophoton_Photon_phi;
  MonitorElement* monophoton_Photon_hovere;
  MonitorElement* monophoton_Photon_sigmaietaieta;
  MonitorElement* monophoton_Photon_trksumptsolidconedr03;
  MonitorElement* monophoton_Photon_e1x5e5x5;
  MonitorElement* monophoton_Photon_e2x5e5x5;
  MonitorElement* monophoton_PFMet;
  MonitorElement* monophoton_PhotonPtOverPFMet;
  MonitorElement* monophoton_PFMet_phi;
  MonitorElement* monophoton_deltaPhiPhotonPFMet;
  MonitorElement* monophoton_PhotonMulti;
  //
  double monophoton_Photon_pt_cut_;
  double monophoton_Photon_met_cut_;
  int monophoton_countPhoton_;

  ///////////////////////////////////
  // Histograms - Displaced Leptons or Jets
  //
  MonitorElement* dispElec_track_effi_lxy;
  MonitorElement* dispElec_elec_effi_lxy;
  MonitorElement* dispMuon_track_effi_lxy;
  MonitorElement* dispMuon_muon_effi_lxy;
  MonitorElement* dispMuon_muonDisp_effi_lxy;
  MonitorElement* dispMuon_muonDispSA_effi_lxy;
  MonitorElement* dispJet_track_effi_lxy;

  double dispFermion_eta_cut_;
  double dispFermion_pt_cut_;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
