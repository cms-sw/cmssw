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

/*   template<class T_u > */
/*     const T_u& down( const edm::Handle<T_u>& up ) const; */
/*     void fetchCollection(edm::Handle<reco::PFCandidateCollection>& c,  */
/* 			 const edm::InputTag& tag,  */
/* 			 const edm::Event& iSetup) const; */


 
  edm::InputTag   inputTagPFCandidates_;
 
  edm::InputTag   inputTagPileUpPFCandidates_;

  edm::InputTag   inputTagIsolatedPFCandidates_;
  
  
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


/* template<class T_u, class T_d> */
/* const T_d& PFTopProjector::down( unsigned uIndex,  */
/* 				 const edm::Handle<T_u>& up, */
/* 				 const edm::Handle<T_d>& d ) const { */

/*   edm::Ref<T_d> up.parent(); */
  
/* /\*   if( parent.productId() == ) *\/ */

/*   return (*d)[0]; */
/* } */


#endif
