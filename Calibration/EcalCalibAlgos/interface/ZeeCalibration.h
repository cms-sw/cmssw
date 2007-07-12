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
// $Id: ZeeCalibration.h,v 1.4 2006/11/20 13:47:56 malgeri Exp $
//
//


// system include files
#include <memory>

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

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

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

  float calculateZMass(const std::pair<calib::CalibElectron*,calib::CalibElectron*>& aZCandidate);

  void bookHistograms();

  // ----------member data ---------------------------
  std::string outputFileName_;
  
  std::string rechitProducer_;
  std::string rechitCollection_;
  std::string scProducer_;
  std::string mcProducer_;
  std::string scCollection_;
  std::string electronProducer_;
  std::string electronCollection_;

  std::string RecalibBarrelHits_;
  
  std::string barrelfile_;

  int read_events;
  
  int loopFlag_;
  
  float calibCoeff[nMaxChannels];
  float initCalibCoeff[nMaxChannels];

  boost::shared_ptr<EcalIntercalibConstants> ical;
  
  ZIterativeAlgorithmWithFit* theAlgorithm_;
  
  // steering parameters
  
  edm::ParameterSet theParameterSet;
  
      
  TH1F* h1_nEleReco_;
  TH1F* h1_recoEleEnergy_;
  
  TH1F* h1_mcEle_Energy_;
  TH1F* h1_mcElePt_;
  TH1F* h1_mcEleEta_;
  TH1F* h1_mcElePhi_;
  TH1F* h1_recoElePt_;
  TH1F* h1_recoEleEta_;
  TH1F* h1_recoElePhi_;
  TH1F* h1_gen_ZMass_;
  TH1F* h1_reco_ZMass_;
  TH1F* h1_reco_ZMassGood_;
  TH1F* h1_reco_ZMassBad_;
  TH1F* h1_ZCandMult_;
  TH1F* h1_RMin_;
  TH1F* h1_RMinZ_;
  TH1F* h1_eleERecoOverEtrue_;
  
  TH2F* h2_coeffVsEta_;
  TH2F* h2_zMassVsLoop_;
  TH2F* h2_zWidthVsLoop_;
  TH2F* h2_coeffVsLoop_;
  TH2F* h2_miscalRecal_;
  TH1F* h1_mc_;
  
  TFile* outputFile_;
      
  unsigned int theMaxLoops;     // Number of loops to loop
  
};
#endif
