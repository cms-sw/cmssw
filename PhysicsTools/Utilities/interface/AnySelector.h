#ifndef Utilities_AnySelector_h
#define Utilities_AnySelector_h
/* \class AnySelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AnySelector.h,v 1.2 2006/11/23 16:03:29 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

template<typename T>
struct AnySelector {
  typedef T value_type;
  AnySelector( const edm::ParameterSet & ) { }
  bool operator()( const T & ) const { return true; }
};

#endif
