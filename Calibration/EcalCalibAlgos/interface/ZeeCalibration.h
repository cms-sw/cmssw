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

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
  virtual void beginOfJob();
  
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
  double getEtaCorrection(const reco::GsfElectron*);

 \
  void fillEleInfo(std::vector<HepMC::GenParticle*>& a, std::map<HepMC::GenParticle*,const reco::GsfElectron*>& b);
  void fillMCInfo(HepMC::GenParticle* mcele);

  void fillMCmap(const std::vector<const reco::GsfElectron*>* electronCollection, const std::vector<HepMC::GenParticle*>& mcEle,std::map<HepMC::GenParticle*,const reco::GsfElectron*>& myMCmap);
  //  void fillMCmap(const reco::ElectronCollection* electronCollection, const std::vector<HepMC::GenParticle*>& mcEle,std::map<HepMC::GenParticle*,const reco::Electron*>& myMCmap);
  
  float EvalDPhi(float Phi,float Phi_ref);
  float EvalDR(float Eta,float Eta_ref,float Phi,float Phi_ref);


  void bookHistograms();

  void resetVariables();

  void resetHistograms();

  void printStatistics();

  std::pair<DetId, double> getHottestDetId(const std::vector<std::pair<DetId, float> >& mySCRecHits, const EBRecHitCollection* ebhits , const EERecHitCollection* eehits);

  bool xtalIsOnModuleBorder( EBDetId myEBDetId );

  float computeCoefficientDistanceAtIteration( float v1[250], float v2[250], int size);

  //  float Calculate_SigmaEtaEta(const reco::SuperCluster &passedCluster);

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
  std::string calibMode_;

  std::string electronProducer_;
  std::string electronCollection_;
  
  std::string RecalibBarrelHits_;
  
  unsigned int etaBins_;
  unsigned int etBins_;

  double etaMin_;
  double etMin_;
  double etaMax_;
  double etMax_;

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
  float NewCalibCoeff[nMaxChannels];
  float calibCoeffError[nMaxChannels];
   float initCalibCoeff[nMaxChannels];

  boost::shared_ptr<EcalIntercalibConstants> ical;
  
  ZIterativeAlgorithmWithFit* theAlgorithm_;

  ZeePlots* myZeePlots_;
  ZeeRescaleFactorPlots* myZeeRescaleFactorPlots_;

  // steering parameters
  
  edm::ParameterSet theParameterSet;

  //  TGraph* graph;

  TH1F* h1_eventsBeforeEWKSelection_;
  TH1F* h1_eventsAfterEWKSelection_;

  TH1F* h1_eventsBeforeBorderSelection_;
  TH1F* h1_eventsAfterBorderSelection_;


  TH2F* h2_fEtaBarrelGood_;
  TH2F* h2_fEtaBarrelBad_;
  TH2F* h2_fEtaEndcapGood_;
  TH2F* h2_fEtaEndcapBad_;
  TH1F* h1_nEleReco_;
  TH1F* h1_eleClasses_;

  TH1F* h_eleEffEta[2];
  TH1F* h_eleEffPhi[2];
  TH1F* h_eleEffPt[2];

  TH1F* h1_seedOverSC_;
  TH1F* h1_preshowerOverSC_;

  TH1F* h1_zMassResol_;
  TH1F* h1_zEtaResol_;
  TH1F* h1_zPhiResol_;
  TH1F* h1_reco_ZMass_;

  TH1F* h1_reco_ZMassCorr_;
  TH1F* h1_reco_ZMassCorrBB_;
  TH1F* h1_reco_ZMassCorrEE_;
  TH1F* h1_reco_ZMassGood_;
  TH1F* h1_reco_ZMassBad_;
  TH1F* h1_ZCandMult_;
  TH1F* h1_RMin_;
  TH1F* h1_RMinZ_;
  TH1F* h1_eleERecoOverEtrue_;

  TH1F* h1_eleEtaResol_;
  TH1F* h1_elePhiResol_;

  TH1F* h_eleEffEta_[2];
  TH1F* h_eleEffPhi_[2];
  TH1F* h_eleEffPt_[2];
  TH1F* h_ESCEtrue_[25];
  TH2F* h_ESCEtrueVsEta_[25];

  TH1F* h_ESCcorrEtrue_[25];
  TH2F* h_ESCcorrEtrueVsEta_[25];

  TH2F* h2_coeffVsEta_;
  TH2F* h2_coeffVsEtaGrouped_;
  TH2F* h2_zMassVsLoop_;
  TH2F* h2_zMassDiffVsLoop_;
  TH2F* h2_zWidthVsLoop_;
  TH2F* h2_coeffVsLoop_;

  TH2F* h2_miscalRecal_;
  //  TH2F* h2_miscalRecalParz_[25];
  TH1F* h1_mc_;
  TH1F* h1_mcParz_[25];
  /*
  TH1F* h_DiffZMassDistr_[25];  
  TH1F* h_ZMassDistr_[25];  
  */
  TH2F* h2_residualSigma_;
  TH2F* h2_miscalRecalEB_;
  //TH2F* h2_miscalRecalEBParz_[25];
  TH1F* h1_mcEB_;
  TH1F* h1_mcEBParz_[25];
  TH2F* h2_miscalRecalEE_;
  //TH2F* h2_miscalRecalEEParz_[25];
  TH1F* h1_mcEE_;
  TH1F* h1_mcEEParz_[25];

  TH2F* h2_chi2_[25];
  TH2F* h2_iterations_[25];

  TH2F * h2_xtalRecalibCoeffBarrel_[25];
  TH2F * h2_xtalRecalibCoeffEndcapMinus_[25];
  TH2F * h2_xtalRecalibCoeffEndcapPlus_[25];
  
  TH2F* h2_xtalMiscalibCoeffBarrel_;
  TH2F* h2_xtalMiscalibCoeffEndcapMinus_;
  TH2F* h2_xtalMiscalibCoeffEndcapPlus_;

  TH1F* h1_weightSumMeanBarrel_;
  TH1F* h1_weightSumMeanEndcap_;

  TH1F* h1_occupancyVsEta_;
  TH1F* h1_occupancyVsEtaGold_;
  TH1F* h1_occupancyVsEtaSilver_;
  TH1F* h1_occupancyVsEtaCrack_;
  TH1F* h1_occupancyVsEtaShower_;
  TH1F* h1_occupancy_;
  TH1F* h1_occupancyBarrel_;
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

  bool              isfirstcall_;
  
};
#endif
