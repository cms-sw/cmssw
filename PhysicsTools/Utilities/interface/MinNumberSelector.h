#ifndef Utilities_MinNumberSelector_h
#define Utilities_MinNumberSelector_h
/* \class SizeMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PtMinSelector.h,v 1.4 2006/10/03 11:44:47 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

struct MinNumberSelector {
  MinNumberSelector( unsigned int minNumber ) : 
    minNumber_( minNumber ) { }
  MinNumberSelector( const edm::ParameterSet & cfg ) : 
    minNumber_( cfg.template getParameter<unsigned int>( "minNumber" ) ) { }
  bool operator()( unsigned int number ) const { return number >= minNumber; }
private:
  unsigned int minNumber_;
};

#endif
