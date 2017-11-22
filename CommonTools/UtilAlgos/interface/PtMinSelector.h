#ifndef UtilAlgos_PtMinSelector_h
#define UtilAlgos_PtMinSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/PtMinSelector.h"

namespace reco {
  namespace modules {

    template<>
    struct ParameterAdapter<PtMinSelector> {
      static PtMinSelector make( const edm::ParameterSet & cfg, edm::ConsumesCollector & iC ) {
	return PtMinSelector( cfg.getParameter<double>( "ptMin" ) );
      }

      // SFINAE trick to provide default when FD::ptMin() is not implemented
      template <typename FD> static decltype(FD::ptMin()) ptMin(int) { return FD::ptMin(); }
      template <typename FD> static auto ptMin(long) { return 0.; }

      template <typename FD>
      static void fillDescriptions(edm::ParameterSetDescription& desc) {
        desc.add<double>("ptMin", ptMin<FD>(0));
      }
    };

  }
}

#endif

