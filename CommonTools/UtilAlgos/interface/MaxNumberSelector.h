#ifndef UtilAlgos_MaxNumberSelector_h
#define UtilAlgos_MaxNumberSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/MaxNumberSelector.h"

namespace reco {
  namespace modules {

    template<>
    struct ParameterAdapter<MaxNumberSelector> {
      static MaxNumberSelector make(const edm::ParameterSet & cfg, edm::ConsumesCollector & iC) {
	return MaxNumberSelector(cfg.getParameter<unsigned int>("maxNumber"));
      }

      // SFINAE trick to provide default when FD::maxNumber() is not implemented
      template <typename FD> static decltype(FD::maxNumber()) maxNumber(int) { return FD::maxNumber(); }
      template <typename FD> static auto maxNumber(long) { return 0U; }

      template <typename FD>
      static void fillDescriptions(edm::ParameterSetDescription& desc) {
        desc.add<unsigned int>("maxNumber", maxNumber<FD>(0));
      }
    };

  }
}

#endif

