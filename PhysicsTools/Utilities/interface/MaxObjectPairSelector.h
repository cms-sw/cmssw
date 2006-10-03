#ifndef RecoAlgos_MaxObjectPairSelector_h
#define RecoAlgos_MaxObjectPairSelector_h
/* \class MaxObjectPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxObjectPairSelector.h,v 1.2 2006/10/03 11:36:10 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

namespace reco {
  namespace helper {
    extern const std::string defaultMaxObjectPairSelectorParamPrefix = "var";
  }
}

template<typename T, typename F, 
	 const std::string & pramPrefix = 
	   reco::helper::defaultMaxObjectPairSelectorParamPrefix>
struct MaxObjectPairSelector {
  typedef T value_type;
  MaxObjectPairSelector( double max ) : 
    max_( max ), fun_() { }
  explicit MaxObjectPairSelector( const edm::ParameterSet & cfg ) : 
    max_( cfg.template getParameter<double>( pramPrefix + "Max" ) ),
    fun_( cfg ) { }
  bool operator()( const value_type & t1, const value_type & t2 ) const { 
    return fun_( t1, t2 ) <= max_;
  }

private:
  double max_;
  F fun_;
};

#endif
