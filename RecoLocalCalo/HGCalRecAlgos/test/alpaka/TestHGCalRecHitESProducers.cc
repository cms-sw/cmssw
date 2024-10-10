// Author: Izaak Neutelings (March 2024)
// Based on: https://github.com/CMS-HGCAL/cmssw/blob/hgcal-condformat-HGCalNANO-13_2_0_pre3_linearity/RecoLocalCalo/HGCalRecAlgos/plugins/alpaka/HGCalRecHitProducer.cc
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include <iomanip>  // for std::setw
#include <future>

// includes for size, calibration, and configuration parameters
#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondFormats/HGCalObjects/interface/HGCalConfiguration.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"  // also for HGCalConfigParamDevice

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class TestHGCalRecHitESProducers : public stream::EDProducer<> {
  public:
    explicit TestHGCalRecHitESProducers(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produce(device::Event&, device::EventSetup const&) override;
    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    edm::ESWatcher<HGCalModuleConfigurationRcd> configWatcher_;
    edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> indexerToken_;
    edm::ESGetToken<HGCalConfiguration, HGCalModuleConfigurationRcd> configToken_;
    device::ESGetToken<hgcalrechit::HGCalConfigParamDevice, HGCalModuleConfigurationRcd> configParamToken_;
    device::ESGetToken<hgcalrechit::HGCalCalibParamDevice, HGCalModuleConfigurationRcd> calibParamToken_;
  };

  TestHGCalRecHitESProducers::TestHGCalRecHitESProducers(const edm::ParameterSet& iConfig) {
    std::cout << "TestHGCalRecHitESProducers::TestHGCalRecHitESProducers" << std::endl;
    indexerToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
    configToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("configSource"));
    configParamToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("configParamSource"));
    calibParamToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("calibParamSource"));
  }

  void TestHGCalRecHitESProducers::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add("indexSource", edm::ESInputTag{})->setComment("Label for module indexer to set SoA size");
    desc.add("configSource", edm::ESInputTag{})->setComment("Label for HGCal configuration for unpacking raw data");
    desc.add("configParamSource", edm::ESInputTag{})
        ->setComment("Label for ROC configuration parameters for calibrations");
    desc.add("calibParamSource", edm::ESInputTag{})->setComment("Label for calibration parameters");
    descriptions.addWithDefaultLabel(desc);
  }

  void TestHGCalRecHitESProducers::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
    std::cout << "TestHGCalRecHitESProducers::beginRun" << std::endl;
  }

  static std::string int2hex(int value) {
    std::stringstream stream;
    stream << "0x" << std::hex << value;
    return stream.str();
  }

  void TestHGCalRecHitESProducers::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    std::cout << "TestHGCalRecHitESProducers::produce" << std::endl;
    auto queue = iEvent.queue();
    auto const& moduleMap = iSetup.getData(indexerToken_);
    auto const& config = iSetup.getData(configToken_);  // HGCalConfiguration
    auto const& configParamDevice = iSetup.getData(configParamToken_);
    //printf("TestHGCalRecHitESProducers::produce: time to load configParamDevice from config ESProducers: %f seconds\n", duration(start,now()));
    auto const& calibParamDevice = iSetup.getData(calibParamToken_);
    //printf("TestHGCalRecHitESProducers::produce: time to load calibParamDevice from calib ESProducers: %f seconds\n", duration(start,now()));

    // Check if there are new conditions and read them
    if (configWatcher_.check(iSetup)) {
      std::cout << "TestHGCalRecHitESProducers::produce: moduleMap.getMaxDataSize()=" << moduleMap.getMaxDataSize()
                << ", moduleMap.getMaxERxSize()=" << moduleMap.getMaxERxSize() << std::endl;

      // ESProducer for global HGCal configuration (structs) with header markers, etc.
      auto nfeds = config.feds.size();  // number of FEDs
      std::cout << "TestHGCalRecHitESProducers::produce: config=" << config << std::endl;
      std::cout << "TestHGCalRecHitESProducers::produce: nfeds=" << nfeds << ", config=" << config << std::endl;
      for (std::size_t fedid = 0; fedid < nfeds; ++fedid) {
        auto fed = config.feds[fedid];   // HGCalFedConfig
        auto nmods = fed.econds.size();  // number of ECON-Ds for this FED
        std::cout << "  fedid=" << fedid << ", nmods=" << nmods << ", passthroughMode=" << fed.mismatchPassthroughMode
                  << ", cbHeaderMarker=0x" << std::hex << fed.cbHeaderMarker << ", slinkHeaderMarker=0x"
                  << fed.slinkHeaderMarker << std::dec << std::endl;
        std::cout << "  modid  nrocs  headerMarker" << std::endl;
        for (std::size_t modid = 0; modid < nmods; ++modid) {
          auto mod = fed.econds[modid];
          auto nrocs = mod.rocs.size();  // number of ECON-Ds for this FED
          std::cout << std::setw(7) << modid << std::setw(7) << nrocs << std::setw(14) << int2hex(mod.headerMarker)
                    << std::endl;
        }
      }

      // Alpaka ESProducer for SoA with configuration parameters with gains
      int size = configParamDevice.view().metadata().size();
      std::cout << "TestHGCalRecHitESProducers::produce: device size=" << size << std::endl;
      std::cout << "  imod  gain" << std::endl;
      for (int imod = 0; imod < size; imod++) {
        if (imod >= 250)
          break;
        std::cout << std::setw(6) << imod << std::setw(6) << uint32_t(configParamDevice.view()[imod].gain())
                  << std::endl;
      }

      // Alpaka ESProducer for SoA with calibration parameters with pedestals, etc.
      size = calibParamDevice.view().metadata().size();
      std::cout << "TestHGCalRecHitESProducers::produce: device size=" << size << std::endl;
      std::cout << "   idx    hex     ADC_ped   CM_slope   CM_ped   BXm1_slope" << std::endl;
      for (int idx = 0; idx < size; idx++) {
        if (idx >= 250)
          break;
        std::cout << std::setw(6) << idx << std::setw(7) << int2hex(idx) << std::dec << std::setw(12)
                  << calibParamDevice.view()[idx].ADC_ped() << std::setw(11) << calibParamDevice.view()[idx].CM_slope()
                  << std::setw(9) << calibParamDevice.view()[idx].CM_ped() << std::setw(13)
                  << calibParamDevice.view()[idx].BXm1_slope() << std::endl;
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define this as a plug-in
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestHGCalRecHitESProducers);
