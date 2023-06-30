#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/HcalObjects/interface/alpaka/HcalMahiPulseOffsetsDevice.h"
#include "CondFormats/HcalObjects/interface/HcalMahiPulseOffsetsSoA.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class HcalMahiPulseOffsetsESProducer : public ESProducer {
  public:
    HcalMahiPulseOffsetsESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      std::vector<int> offsets = iConfig.getParameter<std::vector<int>>("pulseOffsets");

      product = std::make_unique<hcal::HcalMahiPulseOffsetsPortableHost>(offsets.size(), cms::alpakatools::host());

      auto view = product->view();

      for (uint32_t i = 0; i < offsets.size(); i++) {
        view[i] = offsets[i];
      }
      setWhatProduced(this);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::vector<int>>("pulseOffsets", {-3, -2, -1, 0, 1, 2, 3, 4});
      descriptions.addWithDefaultLabel(desc);
    }

    std::shared_ptr<hcal::HcalMahiPulseOffsetsPortableHost> produce(JobConfigurationGPURecord const& iRecord) {
      return product;
    }

  private:
    std::shared_ptr<hcal::HcalMahiPulseOffsetsPortableHost> product;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(HcalMahiPulseOffsetsESProducer);
