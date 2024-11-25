// LST includes
#include "RecoTracker/LSTCore/interface/alpaka/LST.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class LSTModulesDevESProducer : public ESProducer {
  public:
    LSTModulesDevESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) { setWhatProduced(this); }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<lst::LSTESData<DevHost>> produce(TrackerRecoGeometryRecord const& iRecord) {
      return lst::loadAndFillESHost();
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(LSTModulesDevESProducer);
