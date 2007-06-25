#ifndef UtilAlgos_SelectionAdderTrait_h
#define UtilAlgos_SelectionAdderTrait_h
/* \class SelectionAdderTrait
 *
 * \author Luca Lista, INFN
 */
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include <boost/static_assert.hpp>

namespace helper {

  template<typename StoreContainer>
  struct SelectionPointerAdder {
    template<typename C>
    void operator()( StoreContainer & selected, const edm::Handle<C> & c, size_t idx ) {
      selected.push_back( & ( * c )[ idx ] );
    }
  };

  template<typename StoreContainer>
  struct SelectionFirstPointerAdder {
    template<typename C>
    void operator()( StoreContainer & selected, const edm::Handle<C> & c, size_t idx ) {
      selected.push_back( & * ( ( * c )[ idx ].first ) );
    }
  };

  template<typename StoreContainer>
  struct SelectionRefAdder {
    template<typename C>
    void operator()( StoreContainer & selected, const edm::Handle<C> & c, size_t idx ) {
      selected.push_back( edm::Ref<C>( c, idx ) );
    }
  };

  template<typename StoreContainer>
  struct SelectionFirstRefAdder {
    template<typename C>
    void operator()( StoreContainer & selected, const edm::Handle<C> & c, size_t idx ) {
      selected.push_back( ( * c )[ idx ].first );
    }
  };

  template<typename T>
  struct SelectionRefViewAdder {
    void operator()( std::vector<edm::RefToBase<T> > & selected, const edm::Handle<edm::View<T> > & c, size_t idx ) {
      selected.push_back( c->refAt( idx ) );
    }
  };

  template<typename InputCollection, typename StoreContainer>
  struct SelectionAdderTrait {
    BOOST_STATIC_ASSERT(sizeof(InputCollection) == 0); 
  };

  template<typename InputCollection, typename T>
  struct SelectionAdderTrait<InputCollection, std::vector<const T *> > {
    typedef SelectionPointerAdder<std::vector<const T *> > type;
  };

  template<typename R, typename C, typename T>
  struct SelectionAdderTrait<edm::AssociationVector<R, C>, std::vector<const T *> > {
    typedef SelectionFirstPointerAdder<std::vector<const T *> > type;
  };

  template<typename InputCollection, typename C>
  struct SelectionAdderTrait<InputCollection, edm::RefVector<C> > {
    typedef SelectionRefAdder<edm::RefVector<C> > type;
  };

  template<typename R, typename C>
  struct SelectionAdderTrait<edm::AssociationVector<R, C>, edm::RefVector<typename R::product_type> > {
    typedef SelectionFirstRefAdder<edm::RefVector<typename R::product_type> > type;
  };

  template<typename T>
  struct SelectionAdderTrait<edm::View<T>, std::vector<edm::RefToBase<T> > > {
    typedef SelectionRefViewAdder<T> type;
  };

}

#endif
