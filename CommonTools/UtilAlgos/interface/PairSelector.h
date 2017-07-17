#ifndef UtilAlgos_PairSelector_h
#define UtilAlgos_PairSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/PairSelector.h"

namespace reco {
  namespace modules {

    template<typename S1, typename S2>
    struct ParameterAdapter<PairSelector<S1, S2> > {
      static PairSelector<S1, S2> make( const edm::ParameterSet & cfg ) {
	return PairSelector<S1, S2>( modules::make<S1>( cfg ),
				     modules::make<S2>( cfg ) );
      }
      static PairSelector<S1, S2> make( const edm::ParameterSet & cfg, edm::ConsumesCollector & iC ) {
	return PairSelector<S1, S2>( modules::make<S1>( cfg, iC ),
				     modules::make<S2>( cfg, iC ) );
      }
    };

  }
}

#endif

