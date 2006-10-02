#ifndef RecoAlgos_ObjectCountFilter_h
#define RecoAlgos_ObjectCountFilter_h
/** \class ObjectCountFilter
 *
 * Filters an event if a collection has at least N entries
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: ObjectCountFilter.h,v 1.2 2006/09/21 11:56:48 llista Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "PhysicsTools/Utilities/interface/AnySelector.h"
#include <memory>


template<typename S>
class ObjectCountFilterBase : public edm::EDFilter {
public:
  /// constructor 
  explicit ObjectCountFilterBase( const edm::ParameterSet & cfg ) :
    src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
    minNumber_( cfg.template getParameter<unsigned int>( "minNumber" ) ) {
  }
  
protected:
  /// source collection label
  edm::InputTag src_;
  /// minimum number of entries in a collection
  unsigned int minNumber_;
};

template<typename C, typename S = AnySelector<typename C::value_type> >
class ObjectCountFilter : 
  public ObjectCountFilterBase<S> {
public:
  /// constructor 
  explicit ObjectCountFilter( const edm::ParameterSet & cfg ) :
    ObjectCountFilterBase<S>( cfg ),
    select_( cfg ) {
  }
  
private:
  /// process one event
  bool filter( edm::Event& evt, const edm::EventSetup& ) {
    edm::Handle<C> source;
    evt.getByLabel( src_, source );
    size_t n = 0;
    for( typename C::const_iterator i = source->begin(); i != source->end(); ++ i ) {
      if ( select_( * i ) ) n ++;
      if ( n >= minNumber_ ) return true;
    }
    return false;
  }
  /// object filter
  S select_;
};


template<typename C>
class ObjectCountFilter<C, AnySelector<typename C::value_type> > : 
  public ObjectCountFilterBase<AnySelector<typename C::value_type> > {
public:
  /// constructor 
  explicit ObjectCountFilter( const edm::ParameterSet & cfg ) :
    ObjectCountFilterBase<AnySelector<typename C::value_type> >( cfg ) {
  }
  
private:
  /// process one event
  bool filter( edm::Event& evt, const edm::EventSetup& ) {
    edm::Handle<C> source;
    evt.getByLabel( src_, source );
    return source->size() >= minNumber_;
  }
};

#endif
