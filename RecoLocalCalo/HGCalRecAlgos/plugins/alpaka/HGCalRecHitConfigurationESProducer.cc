#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/Math/interface/libminifloat.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h" // depends on HGCalElectronicsMappingRcd
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h" // for HGCalConfigParamHost
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"

#include <string>
#include <iostream> // for std::cout
#include <iomanip> // for std::setw

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcalrechit {

    class HGCalConfigurationESProducer : public ESProducer {
    public:

      HGCalConfigurationESProducer(const edm::ParameterSet& iConfig) : ESProducer(iConfig) {
        if (iConfig.exists("gain"))
          gain_ = iConfig.getParameter<int>("gain");
        auto cc = setWhatProduced(this); //HGCalConfigurationESProducer::produce
        //findingRecord<HGCalModuleConfigurationRcd>();
        //configToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("configSource"));
        //configToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("configSource"));
        indexToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
      }

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::ESInputTag>("indexSource",edm::ESInputTag(""))->setComment("Label for module indexer to set SoA size");
        desc.add<edm::ESInputTag>("configSource",edm::ESInputTag(""))->setComment("Label for ROC configuration parameters");
        desc.addOptional<int>("gain",2)->setComment("Manual override for gain for all modules (1: 80 fC, 2: 160 fC, 4: 320 fC)");
        descriptions.addWithDefaultLabel(desc);
      }

      std::optional<hgcalrechit::HGCalConfigParamHost> produce(const HGCalModuleConfigurationRcd& iRecord) {
        //std::cout << "HGCalConfigurationESProducer::produce" << std::endl;
        //const auto& config = iRecord.get(configToken_);
        //auto const& moduleMap = iRecord.get(indexToken_);
        auto const& moduleMap = iRecord.getRecord<HGCalElectronicsMappingRcd>().get(indexToken_);

        // load dense indexing
        const uint32_t nmod = moduleMap.getMaxERxSize(); // half-ROC-level size
        hgcalrechit::HGCalConfigParamHost product(nmod, cms::alpakatools::host());
        //product.view().map() = moduleMap; // set dense indexing in SoA (now redundant & NOT thread safe !?) 
        std::cout << "HGCalConfigurationESProducer::produce: moduleMap.getMaxDataSize()=" << moduleMap.getMaxDataSize()
                  << ", moduleMap.getMaxERxSize()=" << nmod
                  << ", moduleMap.getMaxModuleSize()=" << moduleMap.getMaxModuleSize() << std::endl;

        // fill SoA with default placeholders
        // TODO: retrieve values from custom JSON format (see HGCalCalibrationESProducer)
        for (uint32_t imod = 0; imod < nmod; imod++) {
          uint8_t gain = gain_; // allow manual override
          product.view()[imod].gain() = gain;
        }

        return product;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> indexToken_;
      //edm::ESGetToken<HGCalCondSerializableConfig, HGCalModuleConfigurationRcd> configToken_;
      int32_t gain_; // manual override of YAML files

    };

  }  // namespace hgcalrechit

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcalrechit::HGCalConfigurationESProducer);
