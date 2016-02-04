#ifndef Demo_PFRootEvent_FWLiteJetProducer_h
#define Demo_PFRootEvent_FWLiteJetProducer_h

#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h" 
#include "RecoParticleFlow/PFRootEvent/interface/CMSIterativeConeAlgorithm.h"
#include "RecoParticleFlow/PFRootEvent/interface/CMSMidpointAlgorithm.h"
#include "RecoParticleFlow/PFRootEvent/interface/FastJetFWLiteWrapper.h"
#include "RecoParticleFlow/PFRootEvent/interface/JetRecoTypes.h"

// #include "FWCore/Framework/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
// #include "DataFormats/Common/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductID.h"

class Utils;
class FWLiteJetProducer{
  /*!
    \author Joanna Weng
    \date July 2006
  */    
 public:
  /// default constructor
  FWLiteJetProducer();  
  /// destructor
  ~FWLiteJetProducer();
  /// Apply Et and E cuts on input object to jet algorihms, prepare 
  /// InputCollection to Jet Algo
  void applyCuts( const reco::CandidatePtrVector& Candidates, 
                  JetReco::InputCollection* input );         
  /// Produce jet collection using CMS Iterative Cone Algorithm 
  void makeIterativeConeJets(const JetReco::InputCollection& fInput, 
                             JetReco::OutputCollection* fOutput);
  /// Produce jet collection using CMS Fast Jet Algorithm       
  void makeFastJets(const JetReco::InputCollection& fInput, 
                    JetReco::OutputCollection* fOutput);
  /// Produce jet collection using CMS Midpoint Jet Algorithm   
  void makeMidpointJets(const JetReco::InputCollection& fInput, 
                        JetReco::OutputCollection* fOutput);
  void print();
  void updateParameter();
  /// Jet Algos --------------------------------------------
  CMSIterativeConeAlgorithm* algoIC_;   
  FastJetFWLiteWrapper algoF_;
  CMSMidpointAlgorithm* algoMC_;

  // Get methods --------------------------------------------
  /// Minimum ET for jet constituents
  double getmEtInputCut(){return mEtInputCut_;}
  /// Minimum E for jet constituents
  double getmEInputCut(){return mEInputCut_;}
  /// Get seed to start jet reconstruction
  double getSeedThreshold(){return seedThreshold_;}
  /// Get radius of the cone
  double  getConeRadius(){return coneRadius_;}
  /// Get fraction of (alowed) overlapping
  double getConeAreaFraction(){return coneAreaFraction_;}
  /// ????
  int getMaxPairSize(){return maxPairSize_;}
  /// ????
  int getMaxIterations(){return maxIterations_;} 
  /// ????
  double getOverlapThreshold(){return overlapThreshold_;}
  /// 
  double getPtMin (){return ptMin_ ;}
  ///
  double getRParam(){return  rparam_;}
  ///
  bool getDebug(){return  debug_;}

  // Set methods --------------------------------------------   
  /// Minimum ET for jet constituents
  void setmEtInputCut (double amEtInputCut){mEtInputCut_=amEtInputCut;}
  /// Minimum E for jet constituents
  void setmEInputCut (double amEInputCut){mEInputCut_=amEInputCut;}
  /// Set seed to start jet reconstruction
  void setSeedThreshold(double aSeedThreshold) {seedThreshold_=aSeedThreshold;}
  /// Set radius of the cone
  void setConeRadius(double aConeRadius) {coneRadius_=aConeRadius;}
  /// Set fraction of (alowed) overlapping
  void setConeAreaFraction(double aConeAreaFraction) {coneAreaFraction_=aConeAreaFraction;}
  /// ????
  void setMaxPairSize(int aMaxPairSize) {maxPairSize_=aMaxPairSize;}
  /// ????
  void setMaxIterations(int amaxIteration) {maxIterations_=amaxIteration;}
  /// ????
  void setOverlapThreshold(double aOverlapThreshold) {overlapThreshold_=aOverlapThreshold;}
  void setPtMin (double aPtMin){ptMin_=aPtMin;}
  void setRParam (double aRparam){rparam_=aRparam;}
  void setDebug(bool aDebug){debug_=aDebug;}  

  // jets parameters ----------------------------------------
 private:       
  /// Minimum ET for jet constituents
  double mEtInputCut_;
  /// Minimum energy for jet constituents
  double mEInputCut_ ;
  /// Seed to start jet reconstruction
  double seedThreshold_;
  /// Radius of the cone
  double coneRadius_;
  /// Fraction of (alowed) overlapping
  double coneAreaFraction_;
  /// ????
  int maxPairSize_;
  /// ????
  int maxIterations_;
  /// ????
  double overlapThreshold_;
  double ptMin_;
  double rparam_;
  bool debug_;
  
};
#endif
