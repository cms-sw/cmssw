#ifndef UtilAlgos_StoreManagerTrait_h
#define UtilAlgos_StoreManagerTrait_h
/* \class helper::CollectionStoreManager
 *
 * \author: Luca Lista, INFN
 *
 */
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include <memory>

namespace helper {

  template<typename Collection>
  struct IteratorToObjectConverter {
    typedef typename Collection::value_type value_type;
    template<typename I>
    static value_type convert( const I & i ) {
      return value_type( * * i );
    }
  };
  
  template<typename T>
   struct IteratorToObjectConverter<edm::OwnVector<T> > {
    typedef std::auto_ptr<T> value_type;
    template<typename I>
    static value_type convert( const I & i ) {
      return value_type( (*i)->clone() );
    }
  };

  template<typename C>
   struct IteratorToObjectConverter<edm::RefVector<C> > {
    typedef edm::Ref<C> value_type;
    template<typename I>
    static value_type convert( const I & i ) {
      return value_type( * i );
    }
  };

  template<typename T>
   struct IteratorToObjectConverter<edm::RefToBaseVector<T> > {
    typedef edm::RefToBase<T> value_type;
    template<typename I>
    static value_type convert( const I & i ) {
      return value_type( * i );
    }
  };

  template<typename OutputCollection, 
	   typename ClonePolicy = IteratorToObjectConverter<OutputCollection> >
  struct CollectionStoreManager {
    typedef OutputCollection collection;
    CollectionStoreManager() : selected_( new collection ) { }
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & ) {
      using namespace std;
      for( I i = begin; i != end; ++ i ) {
	typename ClonePolicy::value_type v = ClonePolicy::convert( i );
        selected_->push_back( v );
      }
    }
    edm::OrphanHandle<collection> put( edm::Event & evt ) {
      return evt.put( selected_ );
    }
    size_t size() const { return selected_->size(); }
  private:
    std::auto_ptr<collection> selected_;
  };

  template<typename OutputCollection>
  struct ObjectSelectorBase : public edm::EDFilter {
    ObjectSelectorBase( const edm::ParameterSet & cfg ) {
      produces<OutputCollection>();
    }    
  };

  template<typename OutputCollection>
  struct StoreManagerTrait {
    typedef CollectionStoreManager<OutputCollection> type;
    typedef ObjectSelectorBase<OutputCollection> base;
  };

}

#endif
