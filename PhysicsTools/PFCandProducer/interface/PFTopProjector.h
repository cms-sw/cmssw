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

#include "DataFormats/Provenance/interface/ProductID.h"

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
 
  /// fills ancestors with ptrs to the PFCandidates that in
  /// one way or another contribute to the candidate pointed to by 
  /// candPtr
  void
    ptrToAncestor( reco::CandidatePtr candRef,
		   reco::CandidatePtrVector& ancestors,
		   const edm::ProductID& ancestorsID ) const;

  /// ancestors is a RefToBase vector. For each object in this vector
  /// get the index and set the corresponding slot to true in the 
  /// masked vector
  void maskAncestors( const reco::CandidatePtrVector& ancestors,
		      std::vector<bool>& masked ) const;
    

  template< class T, class U> 
    void processCollection( const edm::Handle< std::vector<T> >& handle,
			    const edm::Handle< std::vector<U> >& allPFCandidates ,
			    std::vector<bool>& masked,
			    const char* objectName  ) const; 

  template< class T >
    void  printAncestors( const reco::CandidatePtrVector& ancestors,
			  const edm::Handle< std::vector<T> >& allPFCandidates ) const;

  /// ancestor PFCandidates
  edm::InputTag   inputTagPFCandidates_;
 
  /// optional collection of PileUpPFCandidates
  edm::InputTag   inputTagPileUpPFCandidates_;

  /// optional collection of electrons
  edm::InputTag   inputTagIsolatedElectrons_;
  
  /// optional collection of muons
  edm::InputTag   inputTagIsolatedMuons_;
  
  /// optional collection of jets
  edm::InputTag   inputTagPFJets_;

  /// optional collection of taus
  edm::InputTag   inputTagPFTaus_;
  
  /// verbose ?
  bool   verbose_;

  /// label for output PFJet collection
  static const char* pfJetsOutLabel_;

  /// label for output PFCandidates collection
  static const char* pfCandidatesOutLabel_;
  

};

template< class T, class U > 
void PFTopProjector::processCollection( const edm::Handle< std::vector<T> >& handle,
					const edm::Handle< std::vector<U> >& allPFCandidates ,
					std::vector<bool>& masked,
					const char* objectName) const {

  if( handle.isValid() && allPFCandidates.isValid() ) {
    const std::vector<T>& collection = *handle;
    
    if(verbose_) 
      std::cout<<" Collection: "<<objectName
	       <<" size = "<<collection.size()<<std::endl;
    
    for(unsigned i=0; i<collection.size(); i++) {
      
      
      edm::Ptr<T>   ptr( handle, i);
      reco::CandidatePtr basePtr( ptr );
 

      reco::CandidatePtrVector ancestors;
      ptrToAncestor( basePtr,
		     ancestors,
		     allPFCandidates.id() );
      
      if(verbose_) {
/* 	std::cout<<"\t"<<objectName<<" "<<i */
/* 		 <<" pt,eta,phi = " */
/* 		 <<basePtr->pt()<<"," */
/* 		 <<basePtr->eta()<<"," */
/* 		 <<basePtr->phi()<<std::endl; */
	
	std::cout<<"\t"<<collection[i]<<std::endl;
	printAncestors( ancestors, allPFCandidates );
      }
  
      maskAncestors( ancestors, masked );
    }
  }

}


template< class T >
void  PFTopProjector::printAncestors( const reco::CandidatePtrVector& ancestors,
				      const edm::Handle< std::vector<T> >& allPFCandidates ) const {
  
  std::vector<T> pfs = *allPFCandidates;

  for(unsigned i=0; i<ancestors.size(); i++) {

    edm::ProductID id = ancestors[i].id();
    assert( id == allPFCandidates.id() );
 
    unsigned index = ancestors[i].key();
    assert( index < pfs.size() );
    
    std::cout<<"\t\t"<<pfs[index]<<std::endl;
  }
}




#endif
