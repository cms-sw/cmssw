#ifndef RecoAlgos_EtaRangeSelector_h
#define RecoAlgos_EtaRangeSelector_h
/* \class EtaRangeSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PtMinSelector.h,v 1.2 2006/07/25 17:21:31 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

template<typename T>
struct EtaRangeSelector {
  typedef T value_type;
  EtaRangeSelector( double etaMin, double etaMax ) : 
    etaMin_( etaMin ), etaMax_( etaMax ) { }
  EtaRangeSelector( const edm::ParameterSet & cfg ) : 
    etaMin_( cfg.template getParameter<double>( "etaMin" ) ),
    etaMax_( cfg.template getParameter<double>( "etaMax" ) ) {
  }
  bool operator()( const value_type & t ) const { 
    double eta = t.eta();
    return ( eta >= etaMin_ && eta <= etaMax_ ); 
  }
private:
  double etaMin_, etaMax_;
};

#endif
