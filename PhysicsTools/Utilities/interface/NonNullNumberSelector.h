#ifndef Utilities_NonNullNumberSelector_h
#define Utilities_NonNullNumberSelector_h
/* \class SizeMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PtMinSelector.h,v 1.4 2006/10/03 11:44:47 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

struct NonNullNumberSelector {
  NonNullNumberSelector() { }
  NonNullNumberSelector( const edm::ParameterSet & cfg ) { } 
  bool operator()( unsigned int number ) const { return number >= 0; }
};

#endif
