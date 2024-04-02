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
#include "CondFormats/DataRecord/interface/HGCalMappingModuleIndexerRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h" // for HGCalConfigParamHost
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"

#include <string>
#include <iostream> // for std::cout
#include <iomanip> // for std::setw

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcalrechit {

    class HGCalConfigurationESProducer : public ESProducer {
    public:

      HGCalConfigurationESProducer(const edm::ParameterSet& iConfig)
        : ESProducer(iConfig),
          charMode_(iConfig.getParameter<int>("charMode")),
          gain_(iConfig.getParameter<int>("gain")) {
        auto cc = setWhatProduced(this);
        //findingRecord<HGCalModuleConfigurationRcd>();
        //configToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("configSource"));
        moduleIndexerToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("moduleIndexerSource"));
      }

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<int>("charMode",0)->setComment("Manual override for characterization mode to unpack raw data");
        desc.add<int>("gain",2)->setComment("Manual override for gain (1: 80 fC, 2: 160 fC, 4: 320 fC)");
        desc.add<edm::ESInputTag>("moduleIndexerSource",edm::ESInputTag(""))->setComment("Label for module info to set SoA size");
        //desc.add<edm::ESInputTag>("configSource",edm::ESInputTag(""))->setComment("Label for ROC configuration parameters");
        descriptions.addWithDefaultLabel(desc);
      }

      std::optional<hgcalrechit::HGCalConfigParamHost> produce(const HGCalModuleConfigurationRcd& iRecord) {
        //std::cout << "HGCalConfigurationESProducer::produce" << std::endl;
        //const auto& config = iRecord.get(configToken_);
        auto const& moduleMap = iRecord.get(moduleIndexerToken_);

        // load dense indexing
        const uint32_t nmod = moduleMap.getMaxERxSize(); // ROC-level size
        hgcalrechit::HGCalConfigParamHost product(nmod, cms::alpakatools::host());
        //product.view().map() = moduleMap; // set dense indexing in SoA (now redundant & NOT thread safe !?) 
        std::cout << "HGCalConfigurationESProducer::produce: moduleMap.getMaxDataSize()=" << moduleMap.getMaxDataSize()
                  << ", moduleMap.getMaxERxSize()=" << nmod
                  << ", moduleMap.getMaxModuleSize()=" << moduleMap.getMaxModuleSize() << std::endl;

        // NEW: fill SoA with default placeholders
        for (uint32_t imod = 0; imod < nmod; imod++) {
          uint32_t charMode = charMode_;
          uint8_t gain = gain_; // allow manual override
          //std::cout << "Module imod=" << std::setw(3) << imod
          //          << ", charMode=" << charMode << ", gain=" << uint32_t(gain) << std::endl;
          product.view()[imod].gain() = gain;
        }

        //// OLD: fill SoA from YAML
        //// TODO: Use typecode to assign dense index
        //size_t nmods = config.moduleConfigs.size();
        ////LogDebug("HGCalRecHitCalibrationAlgorithms") << "Configuration retrieved for " << nmods << " modules: " << config << std::endl;
        //std::cout << "HGCalRecHitCalibrationAlgorithms" << "Configuration retrieved for " << nmods << " modules: " << config << std::endl;
        //for(auto it : config.moduleConfigs) { // loop over map module electronicsId -> HGCalModuleConfig
        //  HGCalModuleConfig moduleConfig(it.second);
        //  LogDebug("HGCalRecHitCalibrationAlgorithms")
        //    << "Module " << it.first << std::hex << " (0x" << it.first << std::dec
        //    << ") charMode=" << moduleConfig.charMode
        //    << ", ngains=" << moduleConfig.gains.size(); //<< std::endl;
        //  for(auto rocit : moduleConfig.gains) {
        //    uint32_t rocid = rocit.first;
        //    uint8_t gain = (gain_>=1 ? gain_ : rocit.second); // allow manual override
        //    product.view()[cpi.denseROCMap(rocid)].gain() = gain;
        //    LogDebug("HGCalRecHitCalibrationAlgorithms")
        //      << "  ROC " << std::setw(4) << rocid << std::hex << " (0x" << rocid << std::dec
        //      << "): gain=" << (unsigned int) gain << " (override: " << gain_ << ")"; //std::endl;
        //  }
        //}

        return product;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexer, HGCalMappingModuleIndexerRcd> moduleIndexerToken_;
      //edm::ESGetToken<HGCalCondSerializableConfig, HGCalModuleConfigurationRcd> configToken_;
      const int32_t charMode_; // manual override of YAML files
      const int32_t gain_; // manual override of YAML files

    };

  }  // namespace hgcalrechit

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcalrechit::HGCalConfigurationESProducer);
