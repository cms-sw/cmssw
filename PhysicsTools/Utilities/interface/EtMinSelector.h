#ifndef RecoAlgos_EtMinSelector_h
#define RecoAlgos_EtMinSelector_h
/* \class EtMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PtMinSelector.h,v 1.2 2006/07/25 17:21:31 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

template<typename T>
struct EtMinSelector {
  typedef T value_type;
  EtMinSelector( double ptMin ) : 
    etMin_( etMin ) { }
  EtMinSelector( const edm::ParameterSet & cfg ) : 
    etMin_( cfg.template getParameter<double>( "etMin" ) ) { }
  bool operator()( const value_type & t ) const { return t.et() > etMin_; }
private:
  double etMin_;
};

#endif
