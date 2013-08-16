#ifndef CommonTools_UtilAlgos_ObjectCountEventSelector_h
#define CommonTools_UtilAlgos_ObjectCountEventSelector_h

/** \class ObjectCountEventSelector
 *
 * Selects an event if a collection has at least N entries
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: ObjectCountEventSelector.h,v 1.2 2010/02/20 20:55:24 wmtan Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/UtilAlgos/interface/CollectionFilterTrait.h"
#include "CommonTools/UtilAlgos/interface/EventSelectorBase.h"

template<typename C, 
	 typename S = AnySelector,
	 typename N = MinNumberSelector,
         typename CS = typename helper::CollectionFilterTrait<C, S, N>::type>
class ObjectCountEventSelector : public EventSelectorBase
{
 public:
  /// constructor 
  explicit ObjectCountEventSelector( const edm::ParameterSet & cfg ) :
    src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
    select_( reco::modules::make<S>( cfg ) ),
    sizeSelect_( reco::modules::make<N>( cfg ) ) {
  }
 
  bool operator()(edm::Event& evt, const edm::EventSetup&) {
    edm::Handle<C> source;
    evt.getByLabel( src_, source );
    return CS::filter( * source, select_, sizeSelect_ );
  }
 
 private:
  /// source collection label
  edm::InputTag src_;
 
  /// object filter
  S select_;
 
  /// minimum number of entries in a collection
  N sizeSelect_;
};

#endif

