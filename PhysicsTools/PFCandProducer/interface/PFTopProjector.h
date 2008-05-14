#ifndef PhysicsTools_PFCandProducer_PFTopProjector_
#define PhysicsTools_PFCandProducer_PFTopProjector_

// system include files
#include <iostream>
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"


/**\class PFTopProjector 
\brief 

\author Colin Bernet
\date   february 2008
*/




class PFTopProjector : public edm::EDProducer {
 public:

  explicit PFTopProjector(const edm::ParameterSet&);

  ~PFTopProjector();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void beginJob(const edm::EventSetup & c);


 private:
 
  /// fills ancestors with RefToBases to the PFCandidates that in
  /// one way or another contribute to the candidate pointed to by 
  /// candRef
  void
    refToAncestorPFCandidates( reco::CandidateBaseRef candRef,
			       reco::CandidateBaseRefVector& ancestors,
			       const edm::Handle<reco::PFCandidateCollection> allPFCandidates ) 
    const;

  /// ancestors is a RefToBase vector. For each object in this vector
  /// get the index and set the corresponding slot to true in the 
  /// masked vector
  void maskAncestors( const reco::CandidateBaseRefVector& ancestors,
		      std::vector<bool>& masked ) const;
    
  void printAncestors( const reco::CandidateBaseRefVector& ancestors,
		       const edm::Handle<reco::PFCandidateCollection> allPFCandidates ) const;

  /// ancestor PFCandidates
  edm::InputTag   inputTagPFCandidates_;
 
  /// optional collection of PileUpPFCandidates
  edm::InputTag   inputTagPileUpPFCandidates_;

  /// optional collection of IsolatedPFCandidates
  edm::InputTag   inputTagIsolatedPFCandidates_;
  
  /// optional collection of jets
  edm::InputTag   inputTagPFJets_;

  /// optional collection of taus
  edm::InputTag   inputTagPFTaus_;
  
  /// verbose ?
  bool   verbose_;

};

std::ostream& operator<<(std::ostream& out, const reco::PFTau& tau);


#endif
