#ifndef UtilAlgos_SelectionAdderTrait_h
#define UtilAlgos_SelectionAdderTrait_h
/* \class SelectionAdderTrait
 *
 * \author Luca Lista, INFN
 */
#include <vector>
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace edm { class Event; }

namespace helper {

  template<typename SC>
  struct SelectionPointerAdder {
    template<typename C>
    static void add( SC & selected, const edm::Handle<C> & c, size_t idx ) {
      selected.push_back( & ( * c )[ idx ] );
    }
  };

  template<typename SC>
  struct SelectionRefAdder {
    template<typename C>
    static void add( SC & selected, const edm::Handle<C> & c, size_t idx ) {
      selected.push_back( edm::Ref<C>( c, idx ) );
    }
  };

  template<typename S>
  struct SelectionAdderTrait {
    // typedef ... type;
  };

  template<typename T>
  struct SelectionAdderTrait<std::vector<const T *> > {
    typedef SelectionPointerAdder<std::vector<const T *> > type;
  };

  template<typename C>
  struct SelectionAdderTrait<edm::RefVector<C> > {
    typedef SelectionRefAdder<edm::RefVector<C> > type;
  };

}

#endif
