#ifndef CandAlgos_ObjectRefVectorSelector_h
#define CandAlgos_ObjectRefVectorSelector_h
/* \class RefVectorRefVectorStoreMananger
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace helper {
  
  template<typename InputCollection>
  struct RefVectorStoreMananger {
    typedef edm::RefVector<InputCollection> collection;
    RefVectorStoreMananger() : selected_( new collection ) { 
    }
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & ) {
      for( I i = begin; i != end; ++ i )
	selected_->push_back( * i );
    }
    edm::OrphanHandle<collection> put( edm::Event & evt ) {
      return evt.put( selected_ );
    }
    size_t size() const { return selected_->size(); }
  private:
    std::auto_ptr<collection> selected_;
  };
 
}

template<typename Selector, 
	 typename OutputCollection = edm::RefVector<typename Selector::collection>,
	 typename SizeSelector = NonNullNumberSelector,
         typename PostProcessor = reco::helpers::NullPostProcessor<OutputCollection>,
	 typename CollectionStoreManager = helper::RefVectorStoreMananger<typename Selector::collection> >
class ObjectRefVectorSelector : 
  public ObjectSelector<Selector, OutputCollection, SizeSelector, PostProcessor, CollectionStoreManager> {
public:
  explicit ObjectRefVectorSelector( const edm::ParameterSet & cfg ) :
    ObjectSelector<Selector, OutputCollection, SizeSelector, PostProcessor, CollectionStoreManager>( cfg ) { }
};

#endif
