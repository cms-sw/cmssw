#ifndef RecoAlgos_ObjectPairFilter_h
#define RecoAlgos_ObjectPairFilter_h
/** \class ObjectPairFilter
 *
 * Filters an event if one or more pairs of objects passes a given selection
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.4 $
 *
 * $Id: ObjectPairFilter.h,v 1.4 2013/02/28 00:34:12 wmtan Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include <vector>
#include <algorithm>

template<typename C, typename S>
class ObjectPairFilter : public edm::EDFilter {
public:
  /// constructor 
  explicit ObjectPairFilter( const edm::ParameterSet & cfg ) :
    select_( reco::modules::make<S>( cfg ) ),
    src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
    minNumber_( 1 ) {
    std::vector<std::string> ints = cfg.template getParameterNamesForType<unsigned int>();
    const std::string minNumber( "minNumber" );
    bool foundMinNumber = std::find( ints.begin(), ints.end(), minNumber ) != ints.end();
    if ( foundMinNumber )
      minNumber_ = cfg.template getParameter<unsigned int>( minNumber );
  }
 
private:
  /// process one event
  bool filter( edm::Event& evt, const edm::EventSetup&) override {
    edm::Handle<C> source;
    evt.getByLabel( src_, source );
    size_t n = 0;
    for( typename C::const_iterator i = source->begin(); i != source->end(); ++ i )
      for( typename C::const_iterator j = i + 1; j != source->end(); ++ j ) {
	if ( select_( * i, * j ) ) n ++;
	if ( n >= minNumber_ ) return true;
      }
    return false;
  }
  /// object filter
  S select_;
  /// source collection label
  edm::InputTag src_;
  /// minimum number of entries in a collection
  unsigned int minNumber_;
};

#endif

