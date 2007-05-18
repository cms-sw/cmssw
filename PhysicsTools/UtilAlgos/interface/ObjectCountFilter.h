#ifndef RecoAlgos_ObjectCountFilter_h
#define RecoAlgos_ObjectCountFilter_h
/** \class ObjectCountFilter
 *
 * Filters an event if a collection has at least N entries
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.7 $
 *
 * $Id: ObjectCountFilter.h,v 1.7 2007/01/31 14:51:37 llista Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "PhysicsTools/UtilAlgos/interface/AnySelector.h"
#include "PhysicsTools/UtilAlgos/interface/MinNumberSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

namespace helper {
  template<typename C, typename S, typename N>
  struct CollectionSelector {
    static bool filter( const C & source, const S & select, const N & sizeSelect ) {
      size_t n = 0;
      for( typename C::const_iterator i = source.begin(); i != source.end(); ++ i ) {
	if ( select( * i ) ) n ++;
      }
      return sizeSelect( n );      
    }
  };

  template<typename C, typename S>
  struct CollectionSelector<C, S, MinNumberSelector> {
    static bool filter( const C& source, const S & select, const MinNumberSelector & sizeSelect ) {
      size_t n = 0;
      for( typename C::const_iterator i = source.begin(); i != source.end(); ++ i ) {
	if ( select_( * i ) ) n ++;
	if ( sizeSelect( n ) ) return true;
      }
      return false;
    }
  };

  template<typename C, typename N>
  struct CollectionSizeSelector {
    template<typename S>
    static bool filter( const C & source, const S & , const N & sizeSelect ) {
      return sizeSelect( source.size() );
    }
  };

  template<typename C, typename S, typename N>
  struct CollectionSelectorTrait {
    typedef CollectionSelector<C, S, N> type;
  };

  template<typename C, typename N>
  struct CollectionSelectorTrait<C, AnySelector<typename C::value_type>, N> {
    typedef CollectionSizeSelector<C, N> type;
  };

}

template<typename C, 
	 typename S = AnySelector<typename C::value_type>, 
	 typename N = MinNumberSelector,
	 typename CS = typename helper::CollectionSelectorTrait<C, S, N>::type>
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
