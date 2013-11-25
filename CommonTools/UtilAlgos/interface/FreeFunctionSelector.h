#ifndef CommonTools_UtilAlgos_FreeFunctionSelector_h
#define CommonTools_UtilAlgos_FreeFunctionSelector_h
/* \class FreeFunctionSelector
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: FreeFunctionSelector.h,v 1.1 2008/01/22 11:17:58 llista Exp $
 */
template<typename T, bool f(const T&)>
struct FreeFunctionSelector {
  bool operator()(const T& t) {
    return f(t);
  }
};

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {
    template<typename T, bool f(const T&)>
    struct ParameterAdapter<FreeFunctionSelector<T, f> > {
      typedef FreeFunctionSelector<T, f> value_type;
      static value_type make(const edm::ParameterSet & cfg, edm::ConsumesCollector & iC) {
	return value_type();
      }
    };
  }
}



#endif
