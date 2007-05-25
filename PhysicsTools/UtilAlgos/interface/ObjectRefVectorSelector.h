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
  
  template<typename C>
  struct RefVectorStoreMananger {
    RefVectorStoreMananger() : selected_( new edm::RefVector<C> ) { 
    }
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & ) {
      for( I i = begin; i != end; ++ i )
	selected_->push_back( * i );
    }
    void put( edm::Event & evt ) {
      evt.put( selected_ );
    }
    bool empty() const { return selected_->empty(); }
  private:
    std::auto_ptr<edm::RefVector<C> > selected_;
  };

  template<typename C>
  struct RefVectorCollectionStoreManager {
    typedef RefVectorStoreMananger<C> type;
    typedef ObjectSelectorBase<edm::RefVector<C> > base;
  };
 
}

template<typename S, 
	 typename N = NonNullNumberSelector,
         typename P = reco::helpers::NullPostProcessor<typename S::collection>,
	 typename M = typename helper::CollectionStoreManager<typename S::collection>::type, 
	 typename B = typename helper::CollectionStoreManager<typename S::collection>::base>
class ObjectRefVectorSelector : public ObjectSelector<S, N, P, M, B> {
public:
  explicit ObjectRefVectorSelector( const edm::ParameterSet & cfg ) :
    ObjectSelector<S, N, P, M, B>( cfg ) { }
};

#endif
