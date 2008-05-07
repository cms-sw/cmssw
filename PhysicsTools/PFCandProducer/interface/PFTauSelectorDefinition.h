#ifndef PhysicsTools_PFCandProducer_PFTauSelectorDefinition
#define PhysicsTools_PFCandProducer_PFTauSelectorDefinition

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"

#include <iostream>

struct PFTauSelectorDefinition {

  typedef reco::PFTauCollection collection;
  typedef edm::Handle< collection > HandleToCollection;
  typedef std::vector< reco::PFTau *> container;
  typedef container::const_iterator const_iterator;

  PFTauSelectorDefinition ( const edm::ParameterSet & cfg ) :
  discriminatorTag_( cfg.getParameter<edm::InputTag>( "discriminator" ) ) { }

  const_iterator begin() const { return selected_.begin(); }

  const_iterator end() const { return selected_.end(); }

  void select( const HandleToCollection & hc, 
	       const edm::Event & e,
	       const edm::EventSetup& s) {
    selected_.clear();
    
    // collection of PFTaus
    assert( hc.isValid() );

    // associated discriminator
    // does not have to be a discrimination by isolation, 
    // despite the name. 
    edm::Handle<reco::PFTauDiscriminatorByIsolation> 
      hdiscri;
    
    bool found = e.getByLabel(discriminatorTag_,
			      hdiscri);
    
    if(!found) assert(0);
    
    // the discriminator collection and the PFTau collection
    // must have the same size
    assert( hdiscri->size() ==  hc->size());


    unsigned key=0;
    for( collection::const_iterator pftau = hc->begin(); 
         pftau != hc->end(); ++pftau, ++key) {

      PFTauRef pfTauRef(hc, key);
      if( (*hdiscri)[pfTauRef] )
	selected_.push_back( new reco::PFTau(*pftau) );
    }
  }

  size_t size() const { return selected_.size(); }

private:
  container selected_;
  InputTag  discriminatorTag_;
};

#endif
