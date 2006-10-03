#ifndef RecoAlgos_MinObjectPairSelector_h
#define RecoAlgos_MinObjectPairSelector_h
/* \class MinObjectPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinObjectPairSelector.h,v 1.2 2006/10/03 11:36:10 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

namespace reco {
  namespace helper {
    extern const std::string defaultMinObjectPairSelectorParamPrefix = "var";
  }
}

template<typename T, typename F, 
	 const std::string & pramPrefix = 
	   reco::helper::defaultMinObjectPairSelectorParamPrefix>
struct MinObjectPairSelector {
  typedef T value_type;
  MinObjectPairSelector( double min ) : 
    min_( min ),fun_() { }
  explicit MinObjectPairSelector( const edm::ParameterSet & cfg ) : 
    min_( cfg.template getParameter<double>( pramPrefix + "Min" ) ),
    fun_( cfg ) { }
  bool operator()( const value_type & t1, const value_type & t2 ) const { 
    return min_ <= fun_( t1, t2 ); 
  }

private:
  double min_;
  F fun_;
};

#endif
