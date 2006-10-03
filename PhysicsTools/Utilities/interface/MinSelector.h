#ifndef RecoAlgos_MinSelector_h
#define RecoAlgos_MinSelector_h
/* \class MinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinSelector.h,v 1.1 2006/10/03 11:36:29 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

namespace reco {
  namespace helper {
    extern const std::string defaultMinSelectorParamPrefix = "var";
  }
}

template<typename T, double (T::*fun)() const, 
	 const std::string & paramPrefix = 
	   reco::helper::defaultMinSelectorParamPrefix>
struct MinSelector {
  typedef T value_type;
  MinSelector( double min ) : 
    min_( min ) { }
  MinSelector( const edm::ParameterSet & cfg ) : 
    min_( cfg.template getParameter<double>( paramPrefix + "Min" ) ) { }
  bool operator()( const value_type & t ) const { return (t.*fun)() >= min_; }
private:
  double min_;
};

#endif
