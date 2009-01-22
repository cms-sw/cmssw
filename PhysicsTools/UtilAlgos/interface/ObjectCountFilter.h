#ifndef PhysicsTools_UtilAlgos_ObjectCountFilter_h
#define PhysicsTools_UtilAlgos_ObjectCountFilter_h

/** \class ObjectCountFilter
 *
 * Filters an event if a collection has at least N entries
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.10 $
 *
 * $Id: ObjectCountFilter.h,v 1.10 2007/06/18 18:33:52 llista Exp $
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

/*
template<typename C, 
	 typename S = AnySelector,
	 typename N = MinNumberSelector,
	 typename CS = typename helper::CollectionFilterTrait<C, S, N>::type>
class ObjectCountFilter : public edm::EDFilter 
{
 public:
  /// constructor 
  explicit ObjectCountFilter( const edm::ParameterSet & cfg ) :
    eventSelector_( cfg ) {
  }
  
 private:
  /// process one event
  bool filter( edm::Event& evt, const edm::EventSetup& ) {
    return eventSelector_( evt );
  }

  ObjectCountEventSelector<C, S, N, CS> eventSelector_;
};
*/

template<typename C, 
	 typename S = AnySelector,
	 typename N = MinNumberSelector,
	 typename CS = typename helper::CollectionFilterTrait<C, S, N>::type>
struct ObjectCountFilter {
  typedef EventSelectorAdapter< ObjectCountEventSelector<C, S, N, CS> > type;
};

#endif
