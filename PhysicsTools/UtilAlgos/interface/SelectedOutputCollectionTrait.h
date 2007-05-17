#ifndef UtilAlgos_SelectedOutputCollectionTrait_h
#define UtilAlgos_SelectedOutputCollectionTrait_h
/* \class helper SelectedOutputCollection
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Common/interface/AssociationVector.h"

namespace helper {

  template<typename InputCollection>
  struct SelectedOutputCollectionTrait {
    typedef InputCollection type;
  };

  template<typename R, typename C>
  struct SelectedOutputCollectionTrait<edm::AssociationVector<R, C> > {
    typedef typename R::product_type type;
  };

}

#endif
