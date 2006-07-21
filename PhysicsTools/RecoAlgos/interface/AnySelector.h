#ifndef RecoAlgos_AnySelector_h
#define RecoAlgos_AnySelector_h
/* \class AnySelector
 *
 * \author Luca Lista, INFN
 *
 * $Id$
 */
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

template<typename T>
struct AnySelector {
  AnySelector( const edm::ParameterSet & ) { }
  bool operator()( const T & ) { return true; }
};

#endif
