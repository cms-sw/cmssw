#ifndef RecoAlgos_MassRangeSelector_h
#define RecoAlgos_MassRangeSelector_h
/* \class MassRangeSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MassRangeSelector.h,v 1.2 2006/10/03 10:34:03 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

template<typename T>
struct MassRangeSelector {
  typedef T value_type;
  MassRangeSelector( double massMin, double massMax ) : 
    massMin_( massMin ), massMax_( massMax ) { }
  explicit MassRangeSelector( const edm::ParameterSet & cfg ) : 
    massMin_( cfg.template getParameter<double>( "massMin" ) ),
    massMax_( cfg.template getParameter<double>( "massMax" ) ) {
  }
  bool operator()( const value_type & t ) const { 
    double mass = t.mass();
    return ( mass >= massMin_ && mass <= massMax_ ); 
  }
private:
  double massMin_, massMax_;
};

#endif
