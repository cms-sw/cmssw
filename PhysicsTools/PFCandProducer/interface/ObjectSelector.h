#ifndef PhysicsTools_PFCandProducer_ObjectSelector
#define PhysicsTools_PFCandProducer_ObjectSelector

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace ipf2pat {
  
  template< typename Selector, typename CollectionType >
  class ObjectSelector {
  public:
    ObjectSelector(const edm::ParameterSet& ps) : selector_(ps) {}
      
      const CollectionType& select( const edm::Handle<CollectionType>& handleToCollection,
				    const edm::EventBase& event ) {
	/*       static edm::Event e;  */
	static edm::EventSetup s; 
	
	selector_.select( handleToCollection, event, s);
	return selector_.selected();
    }

      
  private:

    Selector selector_;
  }; 

}


#endif 
