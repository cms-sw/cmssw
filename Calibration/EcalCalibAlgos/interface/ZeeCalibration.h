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
// $Id: ZeeCalibration.h,v 1.4 2008/04/29 15:28:56 palmale Exp $
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

#include "Calibration/EcalCalibAlgos/interface/ZeePlots.h"
#include "Calibration/EcalCalibAlgos/interface/ZeeRescaleFactorPlots.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TTree.h"
#include "TFile.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TH1.h"
#include "TH2.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include<vector>
#include<string>

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

 
  void fillEleInfo(std::vector<HepMC::GenParticle*>& a, std::map<HepMC::GenParticle*,const reco::PixelMatchGsfElectron*>& b);
  void fillMCInfo(HepMC::GenParticle* mcele);

  void fillMCmap(const std::vector<const reco::PixelMatchGsfElectron*>* electronCollection, const std::vector<HepMC::GenParticle*>& mcEle,std::map<HepMC::GenParticle*,const reco::PixelMatchGsfElectron*>& myMCmap);
  //  void fillMCmap(const reco::ElectronCollection* electronCollection, const std::vector<HepMC::GenParticle*>& mcElerrel_;
  TH1F* h1_occupancyEndcap_;
  
  TH1F* h1_electronCosTheta_TK_;
  TH1F* h1_electronCosTheta_SC_;
  TH1F* h1_electronCosTheta_SC_TK_;

  TH1F* h1_borderElectronClassification_;


  Int_t BBZN,EBZN,EEZN,BBZN_gg,EBZN_gg,EEZN_gg,BBZN_tt,EBZN_tt,EEZN_tt,BBZN_t0,EBZN_t0,EEZN_t0;
  Int_t NEVT, MCZBB, MCZEB, MCZEE;

  TFile* outputFile_;
      
  unsigned int theMaxLoops;     // Number of loops to loop
 
  bool wantEtaCorrection_;

  unsigned int electronSelection_; 

  double loopArray[50];
  double sigmaArray[50];
  double sigmaErrorArray[50];
  double coefficientDistanceAtIteration[50];

  int BARREL_ELECTRONS_BEFORE_BORDER_CUT;
  int BARREL_ELECTRONS_AFTER_BORDER_CUT;

  int TOTAL_ELECTRONS_IN_BARREL;
  int TOTAL_ELECTRONS_IN_ENDCAP;

  int GOLDEN_ELECTRONS_IN_BARREL;
  int GOLDEN_ELECTRONS_IN_ENDCAP;

  int SILVER_ELECTRONS_IN_BARREL;
  int SILVER_ELECTRONS_IN_ENDCAP;

  int SHOWER_ELECTRONS_IN_BARREL;
  int SHOWER_ELECTRONS_IN_ENDCAP;

  int CRACK_ELECTRONS_IN_BARREL;
  int CRACK_ELECTRONS_IN_ENDCAP;


  edm::InputTag hlTriggerResults_;
  edm::TriggerNames triggerNames_;  // TriggerNames class

  unsigned int  nEvents_;           // number of events processed

  unsigned int  nWasRun_;           // # where at least one HLT was run
  unsigned int  nAccept_;           // # of accepted events
  unsigned int  nErrors_;           // # where at least one HLT had error

  std::vector<unsigned int> hlWasRun_; // # where HLT[i] was run
  std::vector<unsigned int> hlAccept_; // # of events accepted by HLT[i]
  std::vector<unsigned int> hlErrors_; // # of events with error in HLT[i]

  std::vector<std::string>  hlNames_;  // name of each HLT algorithm
  bool init_;                          // vectors initialised or not

  Int_t              triggerCount;
  char              aTriggerNames[200][30];
  bool              aTriggerResults[200];

  Int_t              hltCount;
  char              aHLTNames[6000];
  Int_t              hltNamesLen;
  TString              aNames[200];
  bool              aHLTResults[200];

};
#endif
