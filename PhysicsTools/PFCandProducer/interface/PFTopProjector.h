#ifndef RecoParticleFlow_PFPAT_PFTopProjector_
#define RecoParticleFlow_PFPAT_PFTopProjector_

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
  
  template<class T>
    void fetchCollection(T& c,
			 const edm::InputTag& tag,
			 const edm::Event& iSetup) const;

  const reco::CandidateBaseRef& 
    parent( const reco::CandidateBaseRef& candBaseRef) const;

  reco::CandidateBaseRef
    refToAncestorPFCandidate( reco::CandidateBaseRef candRef,
			      const edm::Handle<reco::PFCandidateCollection> ancestors ) 
    const;
    
  edm::InputTag   inputTagPFCandidates_;
 
  edm::InputTag   inputTagPileUpPFCandidates_;

  edm::InputTag   inputTagIsolatedPFCandidates_;
  
  edm::InputTag   inputTagPFJets_;
  
  
  /// verbose ?
  bool   verbose_;

};


template<class T>
void PFTopProjector::fetchCollection(T& c, 
				     const edm::InputTag& tag, 
				     const edm::Event& iEvent) const {
  
  edm::InputTag empty;
  if( tag==empty ) return;

  bool found = iEvent.getByLabel(tag, c);
  
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get PFCandidates: "
       <<tag<<std::endl;
    edm::LogError("PFCandidates")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }
}



#endif
