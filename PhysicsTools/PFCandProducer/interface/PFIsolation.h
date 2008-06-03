#ifndef PhysicsTools_PFCandProducer_PFIsolation_h_
#define PhysicsTools_PFCandProducer_PFIsolation_h_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

/**\class PFIsolation 
\brief produces IsolatedPFCandidates from PFCandidates

A single isolation algorithm has been implemented up to now. 
This algorithm computes the isolation for all PFCandidates in a collection collection1. 
The isolation is computed with respect to the PFCandidates that are present in a collection collection2. 

The isolation for candidate i in collection1 is defined as sum p_T(j) / p_T(i), where the sum runs on the PFCandidates j in collection 2 that are in a cone of a given delta R (isolation_Cone_DeltaR_) around candidate i. 

If the isolation is greater than isolation_Cone_DeltaR_, an IsolatedPFCandidate is created. 

\todo Implement other isolation algorithms, decouple algos from producers, study all this. 

\author Colin Bernet
\date   february 2008
*/




class PFIsolation : public edm::EDProducer {
 public:

  explicit PFIsolation(const edm::ParameterSet&);

  ~PFIsolation();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void beginJob(const edm::EventSetup & c);

 private:

  double 
    computeIsolation( const reco::PFCandidate& cand, 
		      const reco::PFCandidateCollection& candidates,
		      double isolationCone ) const;
  
  /// PFCandidates for which the isolation will be computed
  edm::InputTag   inputTagPFCandidates_;
  
  /// PFCandidates with which the isolation will be computed
  edm::InputTag   inputTagPFCandidatesForIsolation_;

  /// verbose ?
  bool   verbose_;

  /// min isolation to be considered isolated
  double max_ptFraction_InCone_;
  
  /// isolation cone
  double isolation_Cone_DeltaR_;
  
  /// within the inner cone, PFCandidates are not counted for the isolation.
  /// protects against self isolation
  double isolation_InnerCone_DeltaR_;
};

#endif
