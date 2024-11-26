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

#include "CondFormats/HGCalObjects/interface/HGCalConfiguration.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"      // depends on HGCalElectronicsMappingRcd
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"  // for HGCalConfigParamHost
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"

#include <string>
#include <iostream>  // for std::cout
#include <iomanip>   // for std::setw
//#include <fstream> // needed to read json file with std::ifstream
//#include <nlohmann/json.hpp>
//using json = nlohmann::json;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcalrechit {

    class HGCalConfigurationESProducer : public ESProducer {
    public:
      HGCalConfigurationESProducer(const edm::ParameterSet& iConfig) : ESProducer(iConfig) {
        if (iConfig.exists("gain"))
          gain_ = iConfig.getParameter<int>("gain");
        auto cc = setWhatProduced(this);  //HGCalConfigurationESProducer::produce
        indexToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
        configToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("configSource"));
      }

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::ESInputTag>("indexSource", edm::ESInputTag(""))
            ->setComment("Label for module indexer to set SoA size");
        desc.add<edm::ESInputTag>("configSource", edm::ESInputTag(""))
            ->setComment("Label for ROC configuration parameters");
        desc.addOptional<int>("gain", 2)->setComment(
            "Manual override for gain for all modules (1: 80 fC, 2: 160 fC, 4: 320 fC)");
        descriptions.addWithDefaultLabel(desc);
      }

      std::optional<hgcalrechit::HGCalConfigParamHost> produce(const HGCalModuleConfigurationRcd& iRecord) {
        auto const& config = iRecord.get(configToken_);
        auto const& moduleMap = iRecord.get(indexToken_);

        // load dense indexing
        const uint32_t nERx = moduleMap.getMaxERxSize();  // half-ROC-level size
        hgcalrechit::HGCalConfigParamHost product(nERx, cms::alpakatools::host());
        //std::cout << "HGCalConfigurationESProducer::produce: moduleMap.getMaxDataSize()=" << moduleMap.getMaxDataSize()
        //          << ", moduleMap.getMaxERxSize()=" << nERx
        //          << ", moduleMap.getMaxModuleSize()=" << moduleMap.getMaxModuleSize() << std::endl;

        // fill SoA with gain
        if (gain_ > 0) {  // fill with single value from user override
          //std::cout << "HGCalConfigurationESProducer::produce: fill with default, gain=" << gain_ << std::endl;
          for (uint32_t iroc = 0; iroc < nERx; iroc++) {
            product.view()[iroc].gain() = gain_;
          }
        } else {  // fill with ROC-dependent value from JSON via HGCalConfiguration
          for (uint32_t ifed = 0; ifed < config.feds.size(); ++ifed) {
            for (uint32_t imod = 0; imod < config.feds[ifed].econds.size(); ++imod) {
              for (uint32_t iroc = 0; iroc < config.feds[ifed].econds[imod].rocs.size(); ++iroc) {
                //uint32_t i = getIndexForModuleErx(ifed,imod,iroc); // dense index for ROCs
                uint32_t iroc_dense = moduleMap.getIndexForModuleErx(ifed, imod, iroc);  // dense index for eRx half-ROC
                product.view()[iroc_dense].gain() = config.feds[ifed].econds[imod].rocs[iroc].gain;
              }
            }
          }
        }

        return product;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> indexToken_;
      edm::ESGetToken<HGCalConfiguration, HGCalModuleConfigurationRcd> configToken_;
      int32_t gain_ = -1;  // manual override of YAML files
    };

  }  // namespace hgcalrechit

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcalrechit::HGCalConfigurationESProducer);
