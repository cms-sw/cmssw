#ifndef CommonTools_CandAlgos_CompositeCandSelector_h
#define CommonTools_CandAlgos_CompositeCandSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/CandUtils/interface/CompositeCandSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"

namespace reco {
  namespace modules {

    template<typename Selector, typename T1, typename T2, unsigned int nDau>
      struct ParameterAdapter<CompositeCandSelector<Selector, T1, T2, nDau> > {
	static CompositeCandSelector<Selector, T1, T2, nDau> make(const edm::ParameterSet & cfg) {
	  return CompositeCandSelector<Selector, T1, T2, nDau>(modules::make<Selector>(cfg));
	}
      };

    template<template<typename, typename> class Combiner, typename T1, typename T2, unsigned int nDau>
      struct ParameterAdapter<CompositeCandSelector<Combiner<StringCutObjectSelector<T1>,
                                                             StringCutObjectSelector<T2> >, T1, T2, nDau> > {
	typedef CompositeCandSelector<Combiner<StringCutObjectSelector<T1>,
					       StringCutObjectSelector<T2> >, T1, T2, nDau> Selector;
	  static Selector make(const edm::ParameterSet & cfg, edm::ConsumesCollector & iC) {
	    StringCutObjectSelector<T1> s1(cfg.getParameter<std::string>("daughter1cut"));
	    StringCutObjectSelector<T2> s2(cfg.getParameter<std::string>("daughter2cut"));
	    Combiner<StringCutObjectSelector<T1>, StringCutObjectSelector<T2> > c(s1, s2);
	    return Selector(c);
	  }
      };

  }
}

#endif
