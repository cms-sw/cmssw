#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CondFormats/HcalObjects/interface/alpaka/HcalSiPMCharacteristicsDevice.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristicsSoA.h"
#include "CondFormats/DataRecord/interface/HcalSiPMCharacteristicsRcd.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class HcalSiPMCharacteristicsESProducer : public ESProducer {
  public:
    HcalSiPMCharacteristicsESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      sipmCharacteristicsToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<hcal::HcalSiPMCharacteristicsPortableHost> produce(HcalSiPMCharacteristicsRcd const& iRecord) {
      auto const& sipmCharacteristics = iRecord.get(sipmCharacteristicsToken_);

      size_t const totalItems = sipmCharacteristics.getTypes();

      auto product = std::make_unique<hcal::HcalSiPMCharacteristicsPortableHost>(cms::alpakatools::host(), totalItems);

      auto view = product->view();

      for (uint32_t i = 0; i < sipmCharacteristics.getTypes(); i++) {
        auto vi = view[i];
        auto const type = sipmCharacteristics.getType(i);

        // type index starts with 1 .. 6
        if (static_cast<uint32_t>(type) != i + 1)
          throw cms::Exception("HcalSiPMCharacteristics")
              << "Wrong assumption for HcalSiPMcharacteristics type values, "
              << "should be type value <- type index + 1" << std::endl
              << "Observed type value = " << type << " and index = " << i << std::endl;

        vi.precisionItem() = HcalSiPMCharacteristics::PrecisionItem(type,
                                                                    sipmCharacteristics.getPixels(type),
                                                                    sipmCharacteristics.getNonLinearities(type)[0],
                                                                    sipmCharacteristics.getNonLinearities(type)[1],
                                                                    sipmCharacteristics.getNonLinearities(type)[2],
                                                                    sipmCharacteristics.getCrossTalk(type),
                                                                    sipmCharacteristics.getAuxi1(type),
                                                                    sipmCharacteristics.getAuxi2(type));
      }
      return product;
    }

  private:
    edm::ESGetToken<HcalSiPMCharacteristics, HcalSiPMCharacteristicsRcd> sipmCharacteristicsToken_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(HcalSiPMCharacteristicsESProducer);
