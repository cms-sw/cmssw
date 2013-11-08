#ifndef UtilAlgos_PtMinSelector_h
#define UtilAlgos_PtMinSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/PtMinSelector.h"

namespace reco {
  namespace modules {

    template<>
    struct ParameterAdapter<PtMinSelector> {
      static PtMinSelector make( const edm::ParameterSet & cfg, edm::ConsumesCollector & iC ) {
	return PtMinSelector( cfg.getParameter<double>( "ptMin" ) );
      }
    };

  }
}

#endif

