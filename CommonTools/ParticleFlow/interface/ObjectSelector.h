#ifndef CommonTools_ParticleFlow_ObjectSelector
#define CommonTools_ParticleFlow_ObjectSelector

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"


namespace edm {
  class EventSetup;
}

namespace ipf2pat {
  
  template< typename Selector, typename CollectionType >
  class ObjectSelector {
  public:
    ObjectSelector(const edm::ParameterSet& ps) : 
      eventSetupPtr_(0),
      selector_(ps) {}
      
  
    const CollectionType& select( const edm::Handle<CollectionType>& handleToCollection,
				  const edm::EventBase& event ) {
      /*       static edm::Event e;  */      
      selector_.select( handleToCollection, event, *eventSetupPtr_ );
      return selector_.selected();
    }

      
  private:
    const edm::EventSetup* eventSetupPtr_;

    Selector selector_;
  }; 
  

}



#endif 
