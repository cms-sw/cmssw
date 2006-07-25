#ifndef RecoAlgos_PtMinSelector_h
#define RecoAlgos_PtMinSelector_h
/* \class PtMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PtMinSelector.h,v 1.1 2006/07/25 09:02:56 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

template<typename T>
struct PtMinSelector {
  PtMinSelector( double ptMin ) : 
    ptMin_( ptMin ) { }
  PtMinSelector( const edm::ParameterSet & cfg ) : 
    ptMin_( cfg.template getParameter<double>( "ptMin" ) ) { }
  bool operator()( const T & t ) const { return t.pt() > ptMin_; }
private:
  double ptMin_;
};

#endif
