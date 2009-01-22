#ifndef PhysicsTools_UtilAlgos_ObjectCountFilter_h
#define PhysicsTools_UtilAlgos_ObjectCountFilter_h

/** \class ObjectCountFilter
 *
 * Filters an event if a collection has at least N entries
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.11 $
 *
 * $Id: ObjectCountFilter.h,v 1.11 2009/01/22 13:35:13 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/UtilAlgos/interface/CollectionFilterTrait.h"
#include "PhysicsTools/UtilAlgos/interface/EventSelectorAdapter.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountEventSelector.h"

template<typename C, 
	 typename S = AnySelector,
	 typename N = MinNumberSelector,
	 typename CS = typename helper::CollectionFilterTrait<C, S, N>::type>
struct ObjectCountFilter {
  typedef EventSelectorAdapter< ObjectCountEventSelector<C, S, N, CS> > type;
};

#endif
