#ifndef UtilAlgos_EtaRangeSelector_h
#define UtilAlgos_EtaRangeSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/EtaRangeSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace reco {
  namespace modules {

    template<>
    struct ParameterAdapter<EtaRangeSelector> {
      static EtaRangeSelector make( const edm::ParameterSet & cfg, edm::ConsumesCollector & iC ) {
	return
	  EtaRangeSelector( cfg.getParameter<double>( "etaMin" ),
			    cfg.getParameter<double>( "etaMax" ) );
      }

      static void fillPSetDescription(edm::ParameterSetDescription& desc) {
        desc.add<double>("etaMin", 0.);
        desc.add<double>("etaMax", 0.);
      }
    };

  }
}

#endif

