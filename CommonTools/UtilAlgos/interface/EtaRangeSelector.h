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

      // SFINAE trick to provide default when FD::etaMin() or etaMax() is not implemented
      template <typename FD> static decltype(std::make_pair(FD::etaMin(), FD::etaMax())) etaRange(int) { return std::make_pair(FD::etaMin(), FD::etaMax()); }
      template <typename FD> static auto etaRange(long) { return std::make_pair(0., 0.); }

      template <typename FD>
      static void fillDescriptions(edm::ParameterSetDescription& desc) {
        auto range = etaRange<FD>(0);
        desc.add<double>("etaMin", range.first);
        desc.add<double>("etaMax", range.second);
      }
    };

  }
}

#endif

