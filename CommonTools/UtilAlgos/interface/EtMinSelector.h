#ifndef UtilAlgos_EtMinSelector_h
#define UtilAlgos_EtMinSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/EtMinSelector.h"

namespace reco {
  namespace modules {

    template<>
    struct ParameterAdapter<EtMinSelector> {
      static EtMinSelector make( const edm::ParameterSet & cfg, edm::ConsumesCollector & iC ) {
	return EtMinSelector( cfg.getParameter<double>( "etMin" ) );
      }

      // SFINAE trick to provide default when FD::etMin() is not implemented
      template <typename FD> static decltype(FD::etMin()) ptMin(int) { return FD::etMin(); }
      template <typename FD> static auto etMin(long) { return 0.; }

      template <typename FD>
      static void fillDescriptions(edm::ParameterSetDescription& desc) {
        desc.add<double>("etMin", etMin<FD>(0));
      }
    };

  }
}

#endif

