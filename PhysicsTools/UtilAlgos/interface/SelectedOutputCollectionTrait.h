#ifndef UtilAlgos_SelectedOutputCollectionTrait_h
#define UtilAlgos_SelectedOutputCollectionTrait_h
/* \class helper SelectedOutputCollection
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "FWCore/Framework/interface/View.h"

namespace helper {

  template<typename InputCollection>
  struct SelectedOutputCollectionTrait {
    typedef InputCollection type;
  };

  template<typename R, typename C>
  struct SelectedOutputCollectionTrait<edm::AssociationVector<R, C> > {
    typedef typename R::product_type type;
  };

  template<typename T>
  struct SelectedOutputCollectionTrait<edm::View<T> > {
    typedef typename std::vector<edm::RefToBase<T> > type;
  };

}

#endif
