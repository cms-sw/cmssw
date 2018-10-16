#ifndef UtilAlgos_SelectionAdderTrait_h
#define UtilAlgos_SelectionAdderTrait_h
/* \class SelectionAdderTrait
 *
 * \author Luca Lista, INFN
 */
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/AssociationVector.h"

namespace helper {

  template<typename StoreContainer>
  struct SelectionCopyAdder {
    template<typename C>
    void operator()( StoreContainer & selected, const edm::Handle<C> & c, size_t idx ) {
      selected.push_back( ( * c )[ idx ] );
    }
  };

  template<typename StoreContainer>
  struct SelectionPointerAdder {
    template<typename C>
    void operator()( StoreContainer & selected, const edm::Handle<C> & c, size_t idx ) {
      selected.push_back( & ( * c )[ idx ] );
    }
  };

  template<typename StoreContainer>
  struct SelectionPointerDerefAdder {
    template<typename C>
    void operator()( StoreContainer & selected, const edm::Handle<C> & c, size_t idx ) {
      selected.push_back( & * ( * c )[ idx ] );
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
  struct SelectionFirstRefAdder {
    template<typename C>
    void operator()( StoreContainer & selected, const edm::Handle<C> & c, size_t idx ) {
      selected.push_back( ( * c )[ idx ].first );
    }
  };

  template<typename StoreContainer>
  struct SelectionRefAdder {
    template<typename C>
    void operator()( StoreContainer & selected, const edm::Handle<C> & c, size_t idx ) {
      selected.push_back( edm::Ref<C>( c, idx ) );
    }
  };

  template<typename T>
  struct SelectionRefViewAdder {
    void operator()( edm::RefToBaseVector<T> & selected, const edm::Handle<edm::View<T> > & c, size_t idx ) {
      selected.push_back( c->refAt( idx ) );
    }
  };

  template<typename T>
  struct SelectionPtrViewAdder {
    void operator()( edm::PtrVector<T> & selected, const edm::Handle<edm::View<T> > & c, size_t idx ) {
      selected.push_back( c->ptrAt( idx ) );
    }
  };

  template<typename InputCollection, typename StoreContainer>
  struct SelectionAdderTrait {
    static_assert(sizeof(InputCollection) == 0); 
  };

  template<typename InputCollection, typename T>
  struct SelectionAdderTrait<InputCollection, std::vector<const T *> > {
    typedef SelectionPointerAdder<std::vector<const T *> > type;
  };

  template<typename InputCollection, typename C>
  struct SelectionAdderTrait<InputCollection, edm::RefVector<C> > {
    typedef SelectionRefAdder<edm::RefVector<C> > type;
  };

  template<typename C>
  struct SelectionAdderTrait<edm::RefVector<C>, edm::RefVector<C> > {
    typedef SelectionCopyAdder<edm::RefVector<C> > type;
  };

  template<typename C, typename T>
  struct SelectionAdderTrait<edm::RefVector<C>, std::vector<const T *> > {
    typedef SelectionPointerDerefAdder<std::vector<const T *> > type;
  };

  template<typename T>
  struct SelectionAdderTrait<edm::RefToBaseVector<T>, edm::RefToBaseVector<T> > {
    typedef SelectionCopyAdder<edm::RefToBaseVector<T> > type;
  };

  template<typename T>
  struct SelectionAdderTrait<edm::RefToBaseVector<T>, std::vector<const T *> > {
    typedef SelectionPointerDerefAdder<std::vector<const T *> > type;
  };

  template<typename K, typename C, typename T>
  struct SelectionAdderTrait<edm::AssociationVector<edm::RefProd<K>, C>, std::vector<const T *> > {
    typedef SelectionFirstPointerAdder<std::vector<const T *> > type;
  };

  template<typename C, typename T>
  struct SelectionAdderTrait<edm::AssociationVector<edm::RefToBaseProd<T>, C>, std::vector<const T *> > {
    typedef SelectionFirstPointerAdder<std::vector<const T *> > type;
  };

  template<typename K, typename C>
  struct SelectionAdderTrait<edm::AssociationVector<edm::RefProd<K>, C>, edm::RefVector<K> > {
    typedef SelectionFirstRefAdder<edm::RefVector<K> > type;
  };

  template<typename T, typename C>
  struct SelectionAdderTrait<edm::AssociationVector<edm::RefToBaseProd<T>, C>, 
			     edm::RefToBaseVector<T> > {
    typedef SelectionFirstRefAdder<edm::RefToBaseVector<T> > type;
  };

  template<typename T>
  struct SelectionAdderTrait<edm::View<T>, edm::RefToBaseVector<T> > {
    typedef SelectionRefViewAdder<T> type;
  };

  template<typename T>
  struct SelectionAdderTrait<edm::View<T>, edm::PtrVector<T> > {
    typedef SelectionPtrViewAdder<T> type;
  };

}

#endif

