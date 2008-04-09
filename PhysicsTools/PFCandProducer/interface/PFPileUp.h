#ifndef RecoParticleFlow_PFPAT_PFPileUp_
#define RecoParticleFlow_PFPAT_PFPileUp_

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

/**\class PFPileUp 
\brief produces IsolatedPFCandidates from PFCandidates

\author Colin Bernet
\date   february 2008
*/




class PFPileUp : public edm::EDProducer {
 public:

  explicit PFPileUp(const edm::ParameterSet&);

  ~PFPileUp();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void beginJob(const edm::EventSetup & c);

 private:
  
  void 
    fetchCandidateCollection(edm::Handle<reco::PFCandidateCollection>& c, 
			     const edm::InputTag& tag, 
			     const edm::Event& iSetup) const;


  
  /// PFCandidates in which we'll look for pile up particles 
  edm::InputTag   inputTagPFCandidates_;
  
  /// verbose ?
  bool   verbose_;

};

#endif
