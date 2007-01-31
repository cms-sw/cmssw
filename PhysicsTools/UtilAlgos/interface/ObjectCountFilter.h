#ifndef RecoAlgos_ObjectCountFilter_h
#define RecoAlgos_ObjectCountFilter_h
/** \class ObjectCountFilter
 *
 * Filters an event if a collection has at least N entries
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.6 $
 *
 * $Id: ObjectCountFilter.h,v 1.6 2006/12/07 10:53:14 llista Exp $
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

template<typename C, 
	 typename S = AnySelector<typename C::value_type>, 
	 typename N = MinNumberSelector>
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
    size_t n = 0;
    for( typename C::const_iterator i = source->begin(); i != source->end(); ++ i ) {
      if ( select_( * i ) ) n ++;
      if ( sizeSelect_( n ) ) return true;
    }
    return false;
  }
  /// source collection label
  edm::InputTag src_;
  /// object filter
  S select_;
  /// minimum number of entries in a collection
  N sizeSelect_;
};

template<typename C, typename N>
class ObjectCountFilter<C, AnySelector<typename C::value_type>, N> : public edm::EDFilter {
public:
  /// constructor 
  explicit ObjectCountFilter( const edm::ParameterSet & cfg ) :
    src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
    sizeSelect_( reco::modules::make<N>( cfg ) ) {
  }
  
private:
  /// process one event
  bool filter( edm::Event& evt, const edm::EventSetup& ) {
    edm::Handle<C> source;
    evt.getByLabel( src_, source );
    return sizeSelect_( source->size() );
  }
  /// source collection label
  edm::InputTag src_;
  /// minimum number of entries in a collection
  N sizeSelect_;
};

#endif
