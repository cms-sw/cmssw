#ifndef CALIBRATION_ECALCALIBALGOS_ZEECALIBRATION
#define CALIBRATION_ECALCALIBALGOS_ZEECALIBRATION

// -*- C++ -*-
//
// Package:    ZeeCalibration
// Class:      ZeeCalibration
// 
/**\class ZeeCalibration ZeeCalibration.cc Calibration/EcalCalibAlgos/src/ZeeCalibration.cc

 Description: Perform single electron calibration (tested on TB data only).

 Implementation:
     <Notes on implementation>
*/
//
// $Id: ZeeCalibration.h,v 1.1 2007/07/12 17:27:36 meridian Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <map>

// user include files
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Framework/interface/ESProducerLooper.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Calibration/Tools/interface/ZIterativeAlgorithmWithFit.h"
#include "Calibration/Tools/interface/CalibElectron.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TTree.h"
#include "TFile.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"

// class declaration
//

class ZeeCalibration : public edm::ESProducerLooper {
   
 public:
  
  /// Constructor
  ZeeCalibration( const edm::ParameterSet& iConfig );
  
  /// Destructor
  ~ZeeCalibration();
  
  /// Dummy implementation (job done in duringLoop)
  virtual void produce(edm::Event&, const edm::EventSetup&) {};
  
  /// Called at beginning of job
  virtual void beginOfJob(const edm::EventSetup&);
  
  /// Called at end of job
  virtual void endOfJob();
  
  /// Called at beginning of loop
  virtual void startingNewLoop( unsigned int iLoop );
  
  /// Called at end of loop
  virtual Status endOfLoop( const edm::EventSetup&, unsigned int iLoop );

  /// Called at each event
  virtual Status duringLoop( const edm::Event&, const edm::EventSetup& );
  
  /// Produce Ecal interCalibrations
  virtual boost::shared_ptr<EcalIntercalibConstants> produceEcalIntercalibConstants( const EcalIntercalibConstantsRcd& iRecord );

 private:

/*   ElectronEnergyCorrector myCorrector; */
/*   ElectronClassification myClassificator; */

  double fEtaBarrelBad(double scEta) const;
  double fEtaBarrelGood(double scEta) const;
  double fEtaEndcapBad(double scEta) const;
  double fEtaEndcapGood(double scEta) const;

  int ringNumberCorrector(int k);
  double getEtaCorrection(const reco::PixelMatchGsfElectron*);

  float calculateZMass(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);
  float calculateZEta(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);
  float calculateZTheta(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);
  float calculateZRapidity(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);
  float calculateZPhi(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);
  float calculateZPt(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);
 
  void fillEleInfo(std::vector<HepMC::GenParticle*>& a, std::map<HepMC::GenParticle*,const reco::PixelMatchGsfElectron*>& b);
  void fillMCInfo(HepMC::GenParticle* mcele);

  void fillMCmap(const std::vector<const reco::PixelMatchGsfElectron*>* electronCollection, const std::vector<HepMC::GenParticle*>& mcEle,std::map<HepMC::GenParticle*,const reco::PixelMatchGsfElectron*>& myMCmap);
  //  void fillMCmap(const reco::ElectronCollection* electronCollection, const std::vector<HepMC::GenParticle*>& mcEle,std::map<HepMC::GenParticle*,const reco::Electron*>& myMCmap);
  
  float EvalDPhi(float Phi,float Phi_ref);
  float EvalDR(float Eta,float Eta_ref,float Phi,float Phi_ref);

  float calculateZMassWithCorrectedElectrons(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate, float ele1EnergyCorrection, float ele2EnergyCorrection);

  void bookHistograms();

  // ----------member data ---------------------------
  TTree* myTree;

  std::string outputFileName_;
  
  std::string rechitProducer_;
  std::string rechitCollection_;
  std::string erechitProducer_;
  std::string erechitCollection_;
  std::string scProducer_;
  std::string scCollection_;
 
  std::string scIslandProducer_;
  std::string scIslandCollection_;
  
  std::string mcProducer_;
  
  std::string electronProducer_;
  std::string electronCollection_;
  
  std::string RecalibBarrelHits_;
  
  std::string barrelfile_;
  std::string endcapfile_;

  double minInvMassCut_;
  double maxInvMassCut_;
  double mass; 

  float mass4tree;
  float massDiff4tree;

  int read_events;
  
  int loopFlag_;
  
  float calibCoeff[nMaxChannels];
  float initCalibCoeff[nMaxChannels];

  boost::shared_ptr<EcalIntercalibConstants> ical;
  
  ZIterativeAlgorithmWithFit* theAlgorithm_;
  
  // steering parameters
  
  edm::ParameterSet theParameterSet;
  
  TH2F* h2_fEtaBarrelGood_;
  TH2F* h2_fEtaBarrelBad_;
  TH2F* h2_fEtaEndcapGood_;
  TH2F* h2_fEtaEndcapBad_;
  TH1F* h1_nEleReco_;
  TH1F* h1_eleClasses_;
  TH1F* h1_recoEleEnergy_;
  TH1F* h_eleEffEta[2];
  TH1F* h_eleEffPhi[2];
  TH1F* h_eleEffPt[2];

  TH1F* h1_seedOverSC_;
  TH1F* h1_preshowerOverSC_;

  TH1F* h1_zMassResol_;
  TH1F* h1_zEtaResol_;
  TH1F* h1_zPhiResol_;
  TH1F* h1_mcEle_Energy_;
  TH1F* h1_mcElePt_;
  TH1F* h1_mcEleEta_;
  TH1F* h1_mcElePhi_;
  TH1F* h1_recoElePt_;
  TH1F* h1_recoEleEta_;
  TH1F* h1_recoElePhi_;
  TH1F* h1_reco_ZMass_;
  TH1F* h1_reco_ZEta_;
  TH1F* h1_reco_ZTheta_;
  TH1F* h1_reco_ZRapidity_;
  TH1F* h1_reco_ZPhi_;
  TH1F* h1_reco_ZPt_;
  TH1F* h1_reco_ZMassCorr_;
  TH1F* h1_reco_ZMassCorrBB_;
  TH1F* h1_reco_ZMassCorrEE_;
  TH1F* h1_reco_ZMassGood_;
  TH1F* h1_reco_ZMassBad_;
  TH1F* h1_ZCandMult_;
  TH1F* h1_RMin_;
  TH1F* h1_RMinZ_;
  TH1F* h1_eleERecoOverEtrue_;

  TH1F*  h1_gen_ZMass_;
  TH1F*  h1_gen_ZRapidity_;
  TH1F*  h1_gen_ZEta_;
  TH1F*  h1_gen_ZPhi_;
  TH1F*  h1_gen_ZPt_;
  TH1F* h1_eleEtaResol_;
  TH1F* h1_elePhiResol_;

  TH1F* h_eleEffEta_[2];
  TH1F* h_eleEffPhi_[2];
  TH1F* h_eleEffPt_[2];
  TH1F* h_ESCEtrue_[15];
  TH2F* h_ESCEtrueVsEta_[15];

  TH1F* h_ESCcorrEtrue_[15];
  TH2F* h_ESCcorrEtrueVsEta_[15];

  TH2F* h2_coeffVsEta_;
  TH2F* h2_coeffVsEtaGrouped_;
  TH2F* h2_zMassVsLoop_;
  TH2F* h2_zMassDiffVsLoop_;
  TH2F* h2_zWidthVsLoop_;
  TH2F* h2_coeffVsLoop_;

  TH2F* h2_miscalRecal_;
  TH2F* h2_miscalRecalParz_[15];
  TH1F* h1_mc_;
  TH1F* h1_mcParz_[15];

  TH1F* h_DiffZMassDistr_[15];  
  TH1F* h_ZMassDistr_[15];  

  TH2F* h2_residualSigma_;
  TH2F* h2_miscalRecalEB_;
  TH2F* h2_miscalRecalEBParz_[15];
  TH1F* h1_mcEB_;
  TH1F* h1_mcEBParz_[15];
  TH2F* h2_miscalRecalEE_;
  TH2F* h2_miscalRecalEEParz_[15];
  TH1F* h1_mcEE_;
  TH1F* h1_mcEEParz_[15];

  TH1F* h1_occupancyVsEta_;
  TH1F* h1_occupancyVsEtaGold_;
  TH1F* h1_occupancyVsEtaSilver_;
  TH1F* h1_occupancyVsEtaCrack_;
  TH1F* h1_occupancyVsEtaShower_;
  TH1F* h1_occupancy_;
  
  Int_t BBZN,EBZN,EEZN,BBZN_gg,EBZN_gg,EEZN_gg,BBZN_tt,EBZN_tt,EEZN_tt,BBZN_t0,EBZN_t0,EEZN_t0;
  Int_t NEVT, MCZBB, MCZEB, MCZEE;

  TFile* outputFile_;
      
  unsigned int theMaxLoops;     // Number of loops to loop
 
  bool wantEtaCorrection_;

  unsigned int electronSelection_; 
};
#endif
