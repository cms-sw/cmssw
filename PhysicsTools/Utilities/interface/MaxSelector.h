#ifndef RecoAlgos_MaxSelector_h
#define RecoAlgos_MaxSelector_h
/* \class MaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxSelector.h,v 1.1 2006/10/03 11:45:06 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

namespace reco {
  namespace helper {
    extern const std::string defaultMaxSelectorParamPrefix = "var";
  }
}

template<typename T, double (T::*fun)() const, 
	 const std::string & paramPrefix = 
	   reco::helper::defaultMaxSelectorParamPrefix>
struct MaxSelector {
  typedef T value_type;
  MaxSelector( double max ) : 
    max_( max ) { }
  MaxSelector( const edm::ParameterSet & cfg ) : 
    max_( cfg.template getParameter<double>( paramPrefix + "Max" ) ) { }
  bool operator()( const value_type & t ) const { return (t.*fun)() <= max_; }
private:
  double max_;
};

#endif
