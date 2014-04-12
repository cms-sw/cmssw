#ifndef UtilAlgos_StoreContainerTrait_h
#define UtilAlgos_StoreContainerTrait_h
/* \class helper::StoreContainerTrait
 *
 * \author Luca Lista, INFN
 */
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"

namespace helper {
  template<typename OutputCollection>
    struct StoreContainerTrait {
      typedef std::vector<const typename OutputCollection::value_type *> type;
  };

  template<typename C>
  struct StoreContainerTrait<edm::RefVector<C> > {
    typedef edm::RefVector<C> type;
  };

  template<typename T>
  struct StoreContainerTrait<edm::RefToBaseVector<T> > {
    typedef edm::RefToBaseVector<T> type;
  };

  template<typename T>
  struct StoreContainerTrait<edm::PtrVector<T> > {
    typedef edm::PtrVector<T> type;
  };

  template<typename R, typename C>
   struct StoreContainerTrait<edm::AssociationVector<R, C> > {
     typedef typename StoreContainerTrait<typename R::product_type>::type type;
  };
}

#endif

