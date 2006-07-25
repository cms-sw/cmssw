#ifndef RecoAlgos_AnySelector_h
#define RecoAlgos_AnySelector_h
/* \class AnySelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AnySelector.h,v 1.1 2006/07/21 14:11:26 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

template<typename T>
struct AnySelector {
  AnySelector( const edm::ParameterSet & ) { }
  bool operator()( const T & ) { return true; }
};

#endif
