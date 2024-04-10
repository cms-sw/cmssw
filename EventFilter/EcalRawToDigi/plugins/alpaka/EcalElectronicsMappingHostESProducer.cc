#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalElectronicsMappingDevice.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class EcalElectronicsMappingHostESProducer : public ESProducer {
  public:
    EcalElectronicsMappingHostESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      token_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<EcalElectronicsMappingHost> produce(EcalMappingElectronicsRcd const& iRecord) {
      auto const& mapping = iRecord.get(token_);

      // TODO: 0x3FFFFF * 4B ~= 16MB
      // tmp solution for linear mapping of eid -> did
      int const size = 0x3FFFFF;
      auto product = std::make_unique<EcalElectronicsMappingHost>(size, cms::alpakatools::host());

      // fill in eb
      auto const& barrelValues = mapping.barrelItems();
      for (unsigned int i = 0; i < barrelValues.size(); ++i) {
        EcalElectronicsId eid{barrelValues[i].electronicsid};
        EBDetId did{EBDetId::unhashIndex(i)};
        product->view()[eid.linearIndex()].rawid() = did.rawId();
      }

      // fill in ee
      auto const& endcapValues = mapping.endcapItems();
      for (unsigned int i = 0; i < endcapValues.size(); ++i) {
        EcalElectronicsId eid{endcapValues[i].electronicsid};
        EEDetId did{EEDetId::unhashIndex(i)};
        product->view()[eid.linearIndex()].rawid() = did.rawId();
      }
      return product;
    }

  private:
    edm::ESGetToken<EcalMappingElectronics, EcalMappingElectronicsRcd> token_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(EcalElectronicsMappingHostESProducer);
