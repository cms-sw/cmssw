#ifndef CalibrationIsolatedParticlesIsolatedGenParticles_h
#define CalibrationIsolatedParticlesIsolatedGenParticles_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//TFile Service
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// track associator
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

// ecal / hcal
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

//L1 objects
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "Calibration/IsolatedParticles/interface/GenSimInfo.h"

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1I.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"


namespace{
  class ParticlePtGreater{
  public:
    int operator()(const HepMC::GenParticle * p1, 
		   const HepMC::GenParticle * p2) const{
      return p1->momentum().perp() > p2->momentum().perp();
    }
  };

  class ParticlePGreater{
  public:
    int operator()(const HepMC::GenParticle * p1, 
		   const HepMC::GenParticle * p2) const{
      return p1->momentum().rho() > p2->momentum().rho();
    }
  };
}


class IsolatedGenParticles : public edm::EDAnalyzer {

public:
  explicit IsolatedGenParticles(const edm::ParameterSet&);
  ~IsolatedGenParticles();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  double DeltaPhi(double v1, double v2);
  double DeltaR(double eta1, double phi1, double eta2, double phi2);
  double DeltaR2(double eta1, double phi1, double eta2, double phi2);

  void fillTrack (GlobalPoint& posVec, math::XYZTLorentzVector& momVec, GlobalPoint& posECAL, int pdgId, bool okECAL, bool accpet);
  void fillIsolatedTrack(math::XYZTLorentzVector& momVec, GlobalPoint& posECAL, int pdgId);
  void BookHistograms();
  void clearTreeVectors();
  int  particleCode(int);

  //static const int NPBins   = 21;
  static const int NPBins   = 3;
  static const int NEtaBins = 4;
  static const int Particles=12;
  int    nEventProc;
  double genPartPBins[NPBins+1], genPartEtaBins[NEtaBins+1];
  double pSeed, ptMin, etaMax, pCutIsolate;
  bool   a_Isolation;

  std::string genSrc_;
  const MagneticField *bField;

  bool initL1, useHepMC;
  static const size_t nL1BitsMax=128;
  std::string algoBitToName[nL1BitsMax];
  double a_coneR, a_charIsoR, a_neutIsoR, a_mipR;

  bool   debugL1Info_;
  int    verbosity, debugTrks_;
  bool   printTrkHitPattern_;
  int    myverbose_;
  bool   useJetTrigger_;
  double drLeadJetVeto_, ptMinLeadJet_;
  edm::InputTag L1extraTauJetSource_, L1extraCenJetSource_, L1extraFwdJetSource_;
  edm::InputTag L1extraMuonSource_,   L1extraIsoEmSource_,  L1extraNonIsoEmSource_;
  edm::InputTag L1GTReadoutRcdSource_, L1GTObjectMapRcdSource_;


  edm::Service<TFileService> fs;

  TH1I *h_L1AlgoNames;
  TH1I *h_NEventProc;
  TH2D *h_pEta[Particles];

  TTree *tree;

  std::vector<double> *t_isoTrkPAll;
  std::vector<double> *t_isoTrkPtAll;
  std::vector<double> *t_isoTrkPhiAll;
  std::vector<double> *t_isoTrkEtaAll;
  std::vector<double> *t_isoTrkPdgIdAll;
  std::vector<double> *t_isoTrkDEtaAll;
  std::vector<double> *t_isoTrkDPhiAll;
  
  std::vector<double> *t_isoTrkP;
  std::vector<double> *t_isoTrkPt;
  std::vector<double> *t_isoTrkEne;
  std::vector<double> *t_isoTrkEta;
  std::vector<double> *t_isoTrkPhi;
  std::vector<double> *t_isoTrkEtaEC;
  std::vector<double> *t_isoTrkPhiEC;
  std::vector<double> *t_isoTrkPdgId;

  std::vector<double> *t_maxNearP31x31;
  std::vector<double> *t_cHadronEne31x31, *t_cHadronEne31x31_1, *t_cHadronEne31x31_2, *t_cHadronEne31x31_3;
  std::vector<double> *t_nHadronEne31x31;
  std::vector<double> *t_photonEne31x31;
  std::vector<double> *t_eleEne31x31;
  std::vector<double> *t_muEne31x31;
  
  std::vector<double> *t_maxNearP25x25;
  std::vector<double> *t_cHadronEne25x25, *t_cHadronEne25x25_1, *t_cHadronEne25x25_2, *t_cHadronEne25x25_3;
  std::vector<double> *t_nHadronEne25x25;
  std::vector<double> *t_photonEne25x25;
  std::vector<double> *t_eleEne25x25;
  std::vector<double> *t_muEne25x25;

  std::vector<double> *t_maxNearP21x21;
  std::vector<double> *t_cHadronEne21x21, *t_cHadronEne21x21_1, *t_cHadronEne21x21_2, *t_cHadronEne21x21_3;
  std::vector<double> *t_nHadronEne21x21;
  std::vector<double> *t_photonEne21x21;
  std::vector<double> *t_eleEne21x21;
  std::vector<double> *t_muEne21x21;

  std::vector<double> *t_maxNearP15x15;
  std::vector<double> *t_cHadronEne15x15, *t_cHadronEne15x15_1, *t_cHadronEne15x15_2, *t_cHadronEne15x15_3;
  std::vector<double> *t_nHadronEne15x15;
  std::vector<double> *t_photonEne15x15;
  std::vector<double> *t_eleEne15x15;
  std::vector<double> *t_muEne15x15;
  
  std::vector<double> *t_maxNearP11x11;
  std::vector<double> *t_cHadronEne11x11, *t_cHadronEne11x11_1, *t_cHadronEne11x11_2, *t_cHadronEne11x11_3;
  std::vector<double> *t_nHadronEne11x11;
  std::vector<double> *t_photonEne11x11;
  std::vector<double> *t_eleEne11x11;
  std::vector<double> *t_muEne11x11;

  std::vector<double> *t_maxNearP9x9;
  std::vector<double> *t_cHadronEne9x9, *t_cHadronEne9x9_1, *t_cHadronEne9x9_2, *t_cHadronEne9x9_3;
  std::vector<double> *t_nHadronEne9x9;
  std::vector<double> *t_photonEne9x9;
  std::vector<double> *t_eleEne9x9;
  std::vector<double> *t_muEne9x9;

  std::vector<double> *t_maxNearP7x7;
  std::vector<double> *t_cHadronEne7x7, *t_cHadronEne7x7_1, *t_cHadronEne7x7_2, *t_cHadronEne7x7_3;
  std::vector<double> *t_nHadronEne7x7;
  std::vector<double> *t_photonEne7x7;
  std::vector<double> *t_eleEne7x7;
  std::vector<double> *t_muEne7x7;

  std::vector<double> *t_maxNearP3x3;
  std::vector<double> *t_cHadronEne3x3, *t_cHadronEne3x3_1, *t_cHadronEne3x3_2, *t_cHadronEne3x3_3;
  std::vector<double> *t_nHadronEne3x3;
  std::vector<double> *t_photonEne3x3;
  std::vector<double> *t_eleEne3x3;
  std::vector<double> *t_muEne3x3;

  std::vector<double> *t_maxNearP1x1;
  std::vector<double> *t_cHadronEne1x1, *t_cHadronEne1x1_1, *t_cHadronEne1x1_2, *t_cHadronEne1x1_3;
  std::vector<double> *t_nHadronEne1x1;
  std::vector<double> *t_photonEne1x1;
  std::vector<double> *t_eleEne1x1;
  std::vector<double> *t_muEne1x1;

  std::vector<double> *t_maxNearPHC1x1;
  std::vector<double> *t_cHadronEneHC1x1, *t_cHadronEneHC1x1_1, *t_cHadronEneHC1x1_2, *t_cHadronEneHC1x1_3;
  std::vector<double> *t_nHadronEneHC1x1;
  std::vector<double> *t_photonEneHC1x1;
  std::vector<double> *t_eleEneHC1x1;
  std::vector<double> *t_muEneHC1x1;

  std::vector<double> *t_maxNearPHC3x3;
  std::vector<double> *t_cHadronEneHC3x3, *t_cHadronEneHC3x3_1, *t_cHadronEneHC3x3_2, *t_cHadronEneHC3x3_3;
  std::vector<double> *t_nHadronEneHC3x3;
  std::vector<double> *t_photonEneHC3x3;
  std::vector<double> *t_eleEneHC3x3;
  std::vector<double> *t_muEneHC3x3;

  std::vector<double> *t_maxNearPHC5x5;
  std::vector<double> *t_cHadronEneHC5x5, *t_cHadronEneHC5x5_1, *t_cHadronEneHC5x5_2, *t_cHadronEneHC5x5_3;
  std::vector<double> *t_nHadronEneHC5x5;
  std::vector<double> *t_photonEneHC5x5;
  std::vector<double> *t_eleEneHC5x5;
  std::vector<double> *t_muEneHC5x5;

  std::vector<double> *t_maxNearPHC7x7;
  std::vector<double> *t_cHadronEneHC7x7, *t_cHadronEneHC7x7_1, *t_cHadronEneHC7x7_2, *t_cHadronEneHC7x7_3;
  std::vector<double> *t_nHadronEneHC7x7;
  std::vector<double> *t_photonEneHC7x7;
  std::vector<double> *t_eleEneHC7x7;
  std::vector<double> *t_muEneHC7x7;

  std::vector<double> *t_maxNearPR;
  std::vector<double> *t_cHadronEneR, *t_cHadronEneR_1, *t_cHadronEneR_2, *t_cHadronEneR_3;
  std::vector<double> *t_nHadronEneR;
  std::vector<double> *t_photonEneR;
  std::vector<double> *t_eleEneR;
  std::vector<double> *t_muEneR;

  std::vector<double> *t_maxNearPIsoR;
  std::vector<double> *t_cHadronEneIsoR, *t_cHadronEneIsoR_1, *t_cHadronEneIsoR_2, *t_cHadronEneIsoR_3;
  std::vector<double> *t_nHadronEneIsoR;
  std::vector<double> *t_photonEneIsoR;
  std::vector<double> *t_eleEneIsoR;
  std::vector<double> *t_muEneIsoR;

  std::vector<double> *t_maxNearPHCR;
  std::vector<double> *t_cHadronEneHCR, *t_cHadronEneHCR_1, *t_cHadronEneHCR_2, *t_cHadronEneHCR_3;
  std::vector<double> *t_nHadronEneHCR;
  std::vector<double> *t_photonEneHCR;
  std::vector<double> *t_eleEneHCR;
  std::vector<double> *t_muEneHCR;

  std::vector<double> *t_maxNearPIsoHCR;
  std::vector<double> *t_cHadronEneIsoHCR, *t_cHadronEneIsoHCR_1, *t_cHadronEneIsoHCR_2, *t_cHadronEneIsoHCR_3;
  std::vector<double> *t_nHadronEneIsoHCR;
  std::vector<double> *t_photonEneIsoHCR;
  std::vector<double> *t_eleEneIsoHCR;
  std::vector<double> *t_muEneIsoHCR;

  std::vector<int>    *t_L1Decision;
  std::vector<double> *t_L1CenJetPt,    *t_L1CenJetEta,    *t_L1CenJetPhi;
  std::vector<double> *t_L1FwdJetPt,    *t_L1FwdJetEta,    *t_L1FwdJetPhi;
  std::vector<double> *t_L1TauJetPt,    *t_L1TauJetEta,    *t_L1TauJetPhi;
  std::vector<double> *t_L1MuonPt,      *t_L1MuonEta,      *t_L1MuonPhi;
  std::vector<double> *t_L1IsoEMPt,     *t_L1IsoEMEta,     *t_L1IsoEMPhi;
  std::vector<double> *t_L1NonIsoEMPt,  *t_L1NonIsoEMEta,  *t_L1NonIsoEMPhi;
  std::vector<double> *t_L1METPt,       *t_L1METEta,       *t_L1METPhi;

  spr::genSimInfo isoinfo1x1,   isoinfo3x3,   isoinfo7x7,   isoinfo9x9,   isoinfo11x11;
  spr::genSimInfo isoinfo15x15, isoinfo21x21, isoinfo25x25, isoinfo31x31;
  spr::genSimInfo isoinfoHC1x1, isoinfoHC3x3, isoinfoHC5x5, isoinfoHC7x7;
  spr::genSimInfo isoinfoR,     isoinfoIsoR,  isoinfoHCR,   isoinfoIsoHCR;

};

#endif
