#ifndef RecoAlgos_RangeObjectPairSelector_h
#define RecoAlgos_RangeObjectPairSelector_h
/* \class RangeObjectPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: RangeObjectPairSelector.h,v 1.1 2006/09/20 15:49:36 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

template<typename T, typename F>
struct RangeObjectPairSelector {
  typedef T value_type;
  RangeObjectPairSelector( double min, double max ) : 
    min_( min ), max_( max ), fun_() { }
  explicit RangeObjectPairSelector( const edm::ParameterSet & cfg ) : 
    min_( cfg.template getParameter<double>( "minRange" ) ),
    max_( cfg.template getParameter<double>( "maxRange" ) ),
    fun_( cfg ) { }
  bool operator()( const value_type & t1, const value_type & t2 ) const { 
    double x = fun_( t1, t2 );
    return ( min_ <= x && x <= max_ ); 
  }

private:
  double min_, max_;
  F fun_;
};

#endif
