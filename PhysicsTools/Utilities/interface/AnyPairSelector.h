#ifndef Utilities_AnyPairSelector_h
#define Utilitiess_AnyPairSelector_h
/* \class AnyPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AnySelector.h,v 1.2 2006/11/23 16:03:29 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

template<typename T>
struct AnyPairSelector {
  typedef T value_type;
  AnyPairSelector( const edm::ParameterSet & ) { }
  bool operator()( const T &, const T & ) const { return true; }
};

#endif
