#ifndef RecoAlgos_RangeSelector_h
#define RecoAlgos_RangeSelector_h
/* \class RangeSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: RangeSelector.h,v 1.1 2006/10/03 11:36:29 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

namespace reco {
  namespace helper {
    extern const std::string defaultRangeSelectorParamPrefix = "var";
  }
}

template<typename T, double (T::*fun)() const, 
	 const std::string & paramPrefix = 
	   reco::helper::defaultRangeSelectorParamPrefix>
struct RangeSelector {
  typedef T value_type;
  RangeSelector( double min, double max ) : 
    min_( min ), max_( max ) { }
  RangeSelector( const edm::ParameterSet & cfg ) : 
    min_( cfg.template getParameter<double>( paramPrefix + "Min" ) ),
    max_( cfg.template getParameter<double>( paramPrefix + "Max" ) ) { }
  bool operator()( const value_type & t ) const { 
    double x = (t.*fun)();
    return min_ <= x && x <= max_; 
  }
private:
  double min_, max_;
};

#endif
