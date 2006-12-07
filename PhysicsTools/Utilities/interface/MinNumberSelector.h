#ifndef Utilities_MinNumberSelector_h
#define Utilities_MinNumberSelector_h
/* \class SizeMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinNumberSelector.h,v 1.1 2006/12/07 10:28:31 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <algorithm>

struct MinNumberSelector {
  MinNumberSelector( unsigned int minNumber ) : 
    minNumber_( minNumber ) { }
  MinNumberSelector( const edm::ParameterSet & cfg ) : 
    minNumber_( 1 ) { 
    std::vector<std::string> ints = cfg.template getParameterNamesForType<unsigned int>();
    const std::string minNumber( "minNumber" );
    bool foundMinNumber = std::find( ints.begin(), ints.end(), minNumber ) != ints.end();
    if ( foundMinNumber )
      minNumber_ = cfg.template getParameter<unsigned int>( minNumber );
  }
  bool operator()( unsigned int number ) const { return number >= minNumber_; }

private:
  unsigned int minNumber_;
};

#endif
