#ifndef RecoEgamma_EgammaTools_AnyMVAEstimatorRun2Base_H
#define RecoEgamma_EgammaTools_AnyMVAEstimatorRun2Base_H

#include "DataFormats/Candidate/interface/Candidate.h"

class AnyMVAEstimatorRun2Base {

 public:
  // Constructor, destructor
  AnyMVAEstimatorRun2Base(){};
  ~AnyMVAEstimatorRun2Base(){};

  // Functions that must be provided in derived classes
  // These function should work on electrons or photons
  // of the reco or pat type

  virtual float mvaValue( const edm::Ptr<reco::Candidate>& particle) = 0;
 
  // A specific implementation of MVA is expected to have data members
  // that will contain particle's quantities on which the MVA operates.
  // This function fill their value for a given particle.
  virtual void fillMVAVariables(const edm::Ptr<reco::Candidate>& particle) = 0;
  // A specific implementation of MVA is expected to have one or more categories
  // defined with respect to eta, pt, etc.
  // This function determines the category for a given particle.
  virtual int findCategory( const edm::Ptr<reco::Candidate>& particle) = 0;
 
};


#endif
