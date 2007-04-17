#ifndef RecoAlgos_ObjectCountFilter_h
#define RecoAlgos_ObjectCountFilter_h
/** \class ObjectCountFilter
 *
 * Filters an event if a collection has at least N entries
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.4 $
 *
 * $Id: ObjectCountFilter.h,v 1.4 2006/10/03 09:02:10 llista Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "PhysicsTools/Utilities/interface/AnySelector.h"
#include <algorithm>


template<typename S>
class ObjectCountFilterBase : public edm::EDFilter {
public:
  /// constructor 
  explicit ObjectCountFilterBase( const edm::ParameterSet & cfg ) :
    src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
    minNumber_( 1 ) {
    std::vector<std::string> ints = cfg.template getParameterNamesForType<unsigned int>();
    const std::string minNumber( "minNumber" );
    bool foundMinNumber = std::find( ints.begin(), ints.end(), minNumber ) != ints.end();
    if ( foundMinNumber )
      minNumber_ = cfg.template getParameter<unsigned int>( minNumber );
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
  typedef ObjectCountFilterBase<S> base;
  /// process one event
  bool filter( edm::Event& evt, const edm::EventSetup& ) {
    edm::Handle<C> source;
    evt.getByLabel( base::src_, source );
    size_t n = 0;
    for( typename C::const_iterator i = source->begin(); i != source->end(); ++ i ) {
      if ( select_( * i ) ) n ++;
      if ( n >= base::minNumber_ ) return true;
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
  typedef ObjectCountFilterBase<AnySelector<typename C::value_type> > base;
  /// process one event
  bool filter( edm::Event& evt, const edm::EventSetup& ) {
    edm::Handle<C> source;
    evt.getByLabel( base::src_, source );
    return source->size() >= base::minNumber_;
  }
};

#endif
