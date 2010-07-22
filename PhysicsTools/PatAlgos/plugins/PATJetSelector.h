//
// $Id: PATJetSelector.h,v 1.1 2010/06/16 18:39:13 srappocc Exp $
//

#ifndef PhysicsTools_PatAlgos_PATJetSelector_h
#define PhysicsTools_PatAlgos_PATJetSelector_h

#include "FWCore/Framework/interface/EDFilter.h"

#include "DataFormats/Common/interface/RefVector.h"

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "DataFormats/PatCandidates/interface/Jet.h"


#include <vector>


namespace pat {

  class PATJetSelector : public edm::EDFilter {
  public:


  PATJetSelector( edm::ParameterSet const & params ) : 
    edm::EDFilter( ),
      src_( params.getParameter<edm::InputTag>("src") ),
      cut_( params.getParameter<std::string>("cut") ),
      selector_( cut_ )
      {
	produces< std::vector<pat::Jet> >();
	produces<reco::GenJetCollection> ("genJets");
	produces<CaloTowerCollection > ("caloTowers");
	produces<reco::PFCandidateCollection > ("pfCandidates");
	produces<edm::OwnVector<reco::BaseTagInfo> > ("tagInfos");
      }

    virtual ~PATJetSelector() {}

    virtual void beginJob() {}
    virtual void endJob() {}
    
    virtual bool filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

      std::auto_ptr< std::vector<Jet> > patJets ( new std::vector<Jet>() ); 

      std::auto_ptr<reco::GenJetCollection > genJetsOut ( new reco::GenJetCollection() );
      std::auto_ptr<CaloTowerCollection >  caloTowersOut( new CaloTowerCollection() );
      std::auto_ptr<reco::PFCandidateCollection > pfCandidatesOut( new reco::PFCandidateCollection() );
      std::auto_ptr<edm::OwnVector<reco::BaseTagInfo> > tagInfosOut ( new edm::OwnVector<reco::BaseTagInfo>() );  


      edm::RefProd<reco::GenJetCollection > h_genJetsOut = iEvent.getRefBeforePut<reco::GenJetCollection >( "genJets" );
      edm::RefProd<CaloTowerCollection >  h_caloTowersOut = iEvent.getRefBeforePut<CaloTowerCollection > ( "caloTowers" );
      edm::RefProd<reco::PFCandidateCollection > h_pfCandidatesOut = iEvent.getRefBeforePut<reco::PFCandidateCollection > ( "pfCandidates" );
      edm::RefProd<edm::OwnVector<reco::BaseTagInfo> > h_tagInfosOut = iEvent.getRefBeforePut<edm::OwnVector<reco::BaseTagInfo> > ( "tagInfos" );

      edm::Handle< edm::View<pat::Jet> > h_jets;
      iEvent.getByLabel( src_, h_jets );

      // Loop over the jets
      for ( edm::View<pat::Jet>::const_iterator ibegin = h_jets->begin(),
	      iend = h_jets->end(), ijet = ibegin;
	    ijet != iend; ++ijet ) {

	// Check the selection
	if ( selector_(*ijet) ) {
	  // Add the jets that pass to the output collection
	  patJets->push_back( *ijet );
	  
	  // Copy over the calo towers
	  for ( CaloTowerFwdPtrVector::const_iterator itowerBegin = ijet->caloTowersFwdPtr().begin(),
		  itowerEnd = ijet->caloTowersFwdPtr().end(), itower = itowerBegin;
		itower != itowerEnd; ++itower ) {
	    // Add to global calo tower list
	    caloTowersOut->push_back( **itower );
	    // Update the "forward" bit of the FwdPtr to point at the new tower collection. 

	    //  ptr to "this" tower in the global list	
	    edm::Ptr<CaloTower> outPtr( h_caloTowersOut.id(), 
					&caloTowersOut->back(), 
					caloTowersOut->size() - 1 );     
	    patJets->back().updateFwdCaloTowerFwdPtr( itower - itowerBegin,// index of "this" tower in the jet 
						      outPtr
						      );
	  }

	  
	  // Copy over the pf candidates
	  for ( reco::PFCandidateFwdPtrVector::const_iterator icandBegin = ijet->pfCandidatesFwdPtr().begin(),
		  icandEnd = ijet->pfCandidatesFwdPtr().end(), icand = icandBegin;
		icand != icandEnd; ++icand ) {
	    // Add to global pf candidate list
	    pfCandidatesOut->push_back( **icand );
	    // Update the "forward" bit of the FwdPtr to point at the new tower collection. 

	    // ptr to "this" cand in the global list
	    edm::Ptr<reco::PFCandidate> outPtr( h_pfCandidatesOut.id(), 
						&pfCandidatesOut->back(), 
						pfCandidatesOut->size() - 1 );
	    patJets->back().updateFwdPFCandidateFwdPtr( icand - icandBegin,// index of "this" tower in the jet 
							outPtr
							);
	  }
	  
	  // Copy the tag infos
	  for ( TagInfoFwdPtrCollection::const_iterator iinfoBegin = ijet->tagInfosFwdPtr().begin(),
		  iinfoEnd = ijet->tagInfosFwdPtr().end(), iinfo = iinfoBegin;
		iinfo != iinfoEnd; ++iinfo ) {
	    // Add to global calo tower list
	    tagInfosOut->push_back( **iinfo );
	    // Update the "forward" bit of the FwdPtr to point at the new tower collection. 

	    // ptr to "this" info in the global list
	    edm::Ptr<reco::BaseTagInfo > outPtr( h_tagInfosOut.id(), 
								 &tagInfosOut->back(), 
								 tagInfosOut->size() - 1 );
	    patJets->back().updateFwdTagInfoFwdPtr( iinfo - iinfoBegin,// index of "this" tower in the jet 
						    outPtr
						    );
	  }

	  // Copy the gen jet
	  if ( ijet->genJet() != 0 ) {
	    genJetsOut->push_back( *(ijet->genJet()) );
	    patJets->back().updateFwdGenJetFwdRef( edm::Ref<reco::GenJetCollection>( h_genJetsOut.id(),
										     &genJetsOut->back(),
										     genJetsOut->size() - 1,
										     &*genJetsOut) // ref to "this" genjet in the global list
						   );
	  }

	}
      }

      // put genEvt  in Event
      bool pass = patJets->size() > 0;
      iEvent.put(patJets);

      iEvent.put( genJetsOut, "genJets" );
      iEvent.put( caloTowersOut, "caloTowers" );
      iEvent.put( pfCandidatesOut, "pfCandidates" );
      iEvent.put( tagInfosOut, "tagInfos" );

      return pass;
    }

  protected:
    edm::InputTag                  src_;
    std::string                    cut_;
    StringCutObjectSelector<Jet>   selector_;
  };

}


#endif
