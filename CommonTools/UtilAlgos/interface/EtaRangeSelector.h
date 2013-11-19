#ifndef UtilAlgos_EtaRangeSelector_h
#define UtilAlgos_EtaRangeSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/EtaRangeSelector.h"

namespace reco {
  namespace modules {

    template<>
    struct ParameterAdapter<EtaRangeSelector> {
      static EtaRangeSelector make( const edm::ParameterSet & cfg, edm::ConsumesCollector & iC ) {
	return
	  EtaRangeSelector( cfg.getParameter<double>( "etaMin" ),
			    cfg.getParameter<double>( "etaMax" ) );
      }
    };

  }
}

#endif

