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
    typedef std::unique_ptr<T> value_type;
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

  /*
  template<typename OutputCollection, typename InputCollection>
  struct OutputCollectionCreator {
    static std::unique_ptr<OutputCollection> createNewCollection( const edm::Handle<InputCollection> & ) {
      return std::make_unique<OutputCollection>();
    }
  };

  template<typename T, typename InputCollection>
  struct OutputCollectionCreator<edm::RefToBaseVector<T>, InputCollection> {
    static std::unique_ptr<edm::RefToBaseVector<T> > createNewCollection( const edm::Handle<InputCollection> & h ) {
      return std::make_unique<edm::RefToBaseVector<T> >(h);
    }
  };
  */

  /*
  template<typename T1, typename T2>
  struct OutputCollectionCreator<RefToBaseVector<T1>, RefToBaseVector<T2> > {
    static RefToBaseVector<T1> * createNewCollection( const edm::Handle<RefToBaseVector<T2> > & h ) {
      return new RefToBaseVector<T1>(h);
    }
  };
  */

  template<typename OutputCollection, 
	   typename ClonePolicy = IteratorToObjectConverter<OutputCollection> >
  struct CollectionStoreManager {
    typedef OutputCollection collection;
    template<typename C>
    CollectionStoreManager( const edm::Handle<C> & h ) :
    selected_( new OutputCollection ) { 
      //      selected_ = OutputCollectionCreator<OutputCollection, C>::createNewCollection(h);
    }
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & ) {
      using namespace std;
      for( I i = begin; i != end; ++ i ) {
	typename ClonePolicy::value_type v = ClonePolicy::convert( i );
        selected_->push_back( v );
      }
    }
    edm::OrphanHandle<collection> put(edm::Event & evt) {
      return evt.put(selected_);
    }
    size_t size() const { return selected_->size(); }
  private:
    std::unique_ptr<collection> selected_;
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
