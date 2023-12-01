#ifndef CommonTools_UtilAlgos_CollectionInCollectionFilter_h
#define CommonTools_UtilAlgos_CollectionInCollectionFilter_h

/** \class CollectionInCollectionFilter
 *
 * Filters an event if the total sum of the entries of a collection of collections has at least N entries
 * 
 * \author Marco Musich
 *
 * \version $Revision: 1.1 $
 *
 * $Id: CollectionInCollectionFilter.h,v 1.1 2024/09/23 11:00:00  Exp $
 *
 */

#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/UtilAlgos/interface/CollectionInCollectionFilterTrait.h"
#include "CommonTools/UtilAlgos/interface/EventSelectorAdapter.h"
#include "CommonTools/UtilAlgos/interface/CollectionCountEventSelector.h"

template <typename C,
          typename S = AnySelector,
          typename N = MinNumberSelector,
          typename CS = typename helper::CollectionInCollectionFilterTrait<C, S, N>::type>
struct CollectionInCollectionFilter {
  typedef EventSelectorAdapter<CollectionCountEventSelector<C, S, N, CS> > type;
};

#endif
