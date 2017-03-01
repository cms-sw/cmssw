#ifndef EgammaIsolationProducers_EcalPFClusterIsolation_h
#define EgammaIsolationProducers_EcalPFClusterIsolation_h

//*****************************************************************************
// File:      PFClusterEcalIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matteo Sani
// Institute: UCSD
//*****************************************************************************


//#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/stream/EDProducer.h"

// #include "FWCore/Framework/interface/Event.h"
// #include "FWCore/Framework/interface/MakerMacros.h"
 
// #include "FWCore/ParameterSet/interface/ParameterSet.h"
 
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
 
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateIsolation.h"
 
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include <vector>

template<typename T1>
class EcalPFClusterIsolation {
 public:

  typedef std::vector<T1> T1Collection;
  typedef edm::Ref<T1Collection> T1Ref;

  EcalPFClusterIsolation(double drMax,
			 double drVetoBarrel,
			 double drVetoEndcap,
			 double etaStripBarrel,
			 double etaStripEndcap,
			 double energyBarrel,
			 double energyEndcap);
  
  ~EcalPFClusterIsolation();
  double getSum(T1Ref, edm::Handle<std::vector<reco::PFCluster> >);
  
 private:
  bool computedRVeto(T1Ref candRef, reco::PFClusterRef pfclu);

  double drVeto2_;
  const double drMax_;
  const double drVetoBarrel_;
  const double drVetoEndcap_;
  const double etaStripBarrel_;
  const double etaStripEndcap_;
  const double energyBarrel_;
  const double energyEndcap_;

};

#endif
