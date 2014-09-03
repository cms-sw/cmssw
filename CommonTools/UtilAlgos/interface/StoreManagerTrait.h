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
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include <memory>
#include "boost/static_assert.hpp"
#include "boost/type_traits.hpp"

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


  template<typename T>
  struct IteratorToObjectConverter<edm::PtrVector<T> > {
    typedef edm::Ptr<T> value_type;
    template<typename I>
    static value_type convert( const I & i ) {
      return value_type( * i );
    }
  };



  template<typename OutputCollection, 
	   typename ClonePolicy = IteratorToObjectConverter<OutputCollection> >
  struct CollectionStoreManager {
    typedef OutputCollection collection;
    template<typename C>
    CollectionStoreManager( const edm::Handle<C> & h ) :
      selected_( new OutputCollection ) { 
    }
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

  template<typename OutputCollection, typename EdmFilter>
  struct ObjectSelectorBase : public EdmFilter {
    ObjectSelectorBase( const edm::ParameterSet &) {
      this-> template produces<OutputCollection>();
    }    
  };

  template<typename OutputCollection, typename EdmFilter=edm::EDFilter>
  struct StoreManagerTrait {
    using type = CollectionStoreManager<OutputCollection>;
    using base = ObjectSelectorBase<OutputCollection, EdmFilter>;
  };

}

#endif

