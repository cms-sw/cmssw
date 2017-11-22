#ifndef UtilAlgos_MinNumberSelector_h
#define UtilAlgos_MinNumberSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/MinNumberSelector.h"

namespace reco {
  namespace modules {

    template<>
    struct ParameterAdapter<MinNumberSelector> {
      static MinNumberSelector make(const edm::ParameterSet & cfg, edm::ConsumesCollector & iC) {
	return MinNumberSelector(cfg.getParameter<unsigned int>("minNumber"));
      }

      // SFINAE trick to provide default when FD::minNumber() is not implemented
      template <typename FD> static decltype(FD::minNumber()) minNumber(int) { return FD::minNumber(); }
      template <typename FD> static auto minNumber(long) { return 0U; }

      template <typename FD>
      static void fillDescriptions(edm::ParameterSetDescription& desc) {
        desc.add<unsigned int>("minNumber", minNumber<FD>(0));
      }
    };

  }
}

#endif

