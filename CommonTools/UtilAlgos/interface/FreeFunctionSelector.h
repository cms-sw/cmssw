#ifndef CommonTools_UtilAlgos_FreeFunctionSelector_h
#define CommonTools_UtilAlgos_FreeFunctionSelector_h
/* \class FreeFunctionSelector
 *
 * \author Luca Lista, INFN
 * 
 * \version $Id: FreeFunctionSelector.h,v 1.1 2009/03/03 13:07:26 llista Exp $  
 */
template<typename T, bool f(const T&)>
struct FreeFunctionSelector {
  bool operator()(const T& t) {
    return f(t);
  }
};

#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {
    template<typename T, bool f(const T&)>
    struct ParameterAdapter<FreeFunctionSelector<T, f> > { 
      typedef FreeFunctionSelector<T, f> value_type;
      static value_type make(const edm::ParameterSet & cfg) {
	return value_type();
      }
    };
  }
}



#endif
