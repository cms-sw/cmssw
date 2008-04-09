#ifndef RecoParticleFlow_PFPAT_PFIsolation_h_
#define RecoParticleFlow_PFPAT_PFIsolation_h_

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
  
  void 
    fetchCandidateCollection(edm::Handle<reco::PFCandidateCollection>& c, 
			     const edm::InputTag& tag, 
			     const edm::Event& iSetup) const;


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

};

#endif
