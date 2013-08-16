#ifndef CommonTools_UtilAlgos_ObjectCountFilter_h
#define CommonTools_UtilAlgos_ObjectCountFilter_h

/** \class ObjectCountFilter
 *
 * Filters an event if a collection has at least N entries
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: ObjectCountFilter.h,v 1.2 2009/04/08 11:35:00 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/UtilAlgos/interface/CollectionFilterTrait.h"
#include "CommonTools/UtilAlgos/interface/EventSelectorAdapter.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountEventSelector.h"

template<typename C, 
	 typename S = AnySelector,
	 typename N = MinNumberSelector,
	 typename CS = typename helper::CollectionFilterTrait<C, S, N>::type>
struct ObjectCountFilter {
  typedef EventSelectorAdapter< ObjectCountEventSelector<C, S, N, CS> > type;
};

#endif
