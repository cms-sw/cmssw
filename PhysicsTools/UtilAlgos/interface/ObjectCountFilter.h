#ifndef RecoAlgos_ObjectCountFilter_h
#define RecoAlgos_ObjectCountFilter_h
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

template<typename C, 
	 typename S = AnySelector,
	 typename N = MinNumberSelector,
	 typename CS = typename helper::CollectionFilterTrait<C, S, N>::type>
class ObjectCountFilter : public edm::EDFilter {
public:
  /// constructor 
  explicit ObjectCountFilter( const edm::ParameterSet & cfg ) :
    src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
    select_( reco::modules::make<S>( cfg ) ),
    sizeSelect_( reco::modules::make<N>( cfg ) ) {
  }
  
private:
  /// process one event
  bool filter( edm::Event& evt, const edm::EventSetup& ) {
    edm::Handle<C> source;
    evt.getByLabel( src_, source );
    return CS::filter( * source, select_, sizeSelect_ );
  }
  /// source collection label
  edm::InputTag src_;
  /// object filter
  S select_;
  /// minimum number of entries in a collection
  N sizeSelect_;
};

#endif
