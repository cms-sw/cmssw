#ifndef UtilAlgos_SelectedOutputCollectionTrait_h
#define UtilAlgos_SelectedOutputCollectionTrait_h
/* \class helper SelectedOutputCollection
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/View.h"

namespace helper {

  template<typename InputCollection>
  struct SelectedOutputCollectionTrait {
    typedef InputCollection type;
  };

  template<typename K, typename C>
  struct SelectedOutputCollectionTrait<edm::AssociationVector<edm::RefProd<K>, C> > {
    typedef typename edm::RefProd<K>::product_type type;
  };

  template<typename T, typename C>
  struct SelectedOutputCollectionTrait<edm::AssociationVector<edm::RefToBaseProd<T>, C> > {
    typedef typename edm::RefToBaseVector<T> type;
  };

  template<typename T>
  struct SelectedOutputCollectionTrait<edm::View<T> > {
    typedef typename edm::RefToBaseVector<T> type;
  };

  template<typename T>
  struct SelectedOutputCollectionTrait<edm::RefToBaseVector<T> > {
    typedef typename edm::RefToBaseVector<T> type;
  };

  template<typename C>
  struct SelectedOutputCollectionTrait<edm::RefVector<C> > {
    typedef typename edm::RefVector<C> type;
  };

}

#endif

