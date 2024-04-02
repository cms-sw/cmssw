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
#include <iomanip> // for std::setw
#include <future>

// includes for size, calibration, and configuration parameters
#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/DataRecord/interface/HGCalMappingModuleIndexerRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h" // also for HGCalConfigParamDevice

//template<class T> double duration(T t0,T t1) {
//  auto elapsed_secs = t1-t0;
//  typedef std::chrono::duration<float> float_seconds;
//  auto secs = std::chrono::duration_cast<float_seconds>(elapsed_secs);
//  return secs.count();
//}
//
//typedef std::chrono::time_point<std::chrono::steady_clock> time_t_;
//inline time_t_ now() {
//  return std::chrono::steady_clock::now();
//}

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
    edm::ESWatcher<HGCalMappingModuleIndexerRcd> calibWatcher_;
    edm::ESGetToken<HGCalMappingModuleIndexer, HGCalMappingModuleIndexerRcd> moduleIndexerToken_;
    device::ESGetToken<hgcalrechit::HGCalConfigParamDevice, HGCalModuleConfigurationRcd> configToken_;
    device::ESGetToken<hgcalrechit::HGCalCalibParamDevice, HGCalMappingModuleIndexerRcd> calibToken_;
  };

  TestHGCalRecHitESProducers::TestHGCalRecHitESProducers(const edm::ParameterSet& iConfig) {
    std::cout << "TestHGCalRecHitESProducers::TestHGCalRecHitESProducers" << std::endl;
    configToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("configSource"));
    calibToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("calibSource"));
    moduleIndexerToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("calibSource"));
  }

  void TestHGCalRecHitESProducers::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup){
    std::cout << "TestHGCalRecHitESProducers::beginRun" << std::endl;
  }

  void TestHGCalRecHitESProducers::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    std::cout << "TestHGCalRecHitESProducers::produce" << std::endl;
    auto queue = iEvent.queue();
    auto const& moduleMap = iSetup.getData(moduleIndexerToken_);
    auto const& configParamDevice = iSetup.getData(configToken_);
    //printf("TestHGCalRecHitESProducers::produce: time to load configParamDevice from config ESProducers: %f seconds\n", duration(start,now()));
    auto const& calibParamDevice = iSetup.getData(calibToken_);
    //printf("TestHGCalRecHitESProducers::produce: time to load calibParamDevice from calib ESProducers: %f seconds\n", duration(start,now()));

    // Check if there are new conditions and read them
    if (configWatcher_.check(iSetup)){
      int size = configParamDevice.view().metadata().size();
      std::cout << "TestHGCalRecHitESProducers::produce: moduleMap.getMaxDataSize()=" << moduleMap.getMaxDataSize()
                << ", moduleMap.getMaxERxSize()=" << moduleMap.getMaxERxSize() << std::endl;
      std::cout << "TestHGCalRecHitESProducers::produce: device size=" << size << std::endl;
      std::cout << "  imod  gain" << std::endl;
      for(int imod=0; imod<size; imod++) {
        if(imod>=250) break;
        std::cout << std::setw(6) << imod
          << std::setw(6) << uint32_t(configParamDevice.view()[imod].gain()) << std::endl;
      }
    }

    // Check if there are new conditions and read them
    if(calibWatcher_.check(iSetup)){
      int size = calibParamDevice.view().metadata().size();
      std::cout << "TestHGCalRecHitESProducers::produce: moduleMap.getMaxDataSize()=" << moduleMap.getMaxDataSize()
                << ", moduleMap.getMaxERxSize()=" << moduleMap.getMaxERxSize() << std::endl;
      std::cout << "TestHGCalRecHitESProducers::produce: device size=" << size << std::endl;
      std::cout << "  idx  hex  ADC_ped  CM_slope  CM_ped  BXm1_slope" << std::endl;
      for(int idx=0; idx<size; idx++) {
        if(idx>=250) break;
        std::cout << std::setw(5) << idx << std::hex << std::setw(5) << idx << std::dec
          << std::setw(9)  << calibParamDevice.view()[idx].ADC_ped()
          << std::setw(10) << calibParamDevice.view()[idx].CM_slope()
          << std::setw(8)  << calibParamDevice.view()[idx].CM_ped()
          << std::setw(12) << calibParamDevice.view()[idx].BXm1_slope()
          // << std::setw(6) << calibParamDevice.view()[idx].BXm1_offset() // redundant
          << std::endl;
      }
    }

  }

  void TestHGCalRecHitESProducers::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add("configSource", edm::ESInputTag{})->setComment("Label for ROC configuration parameters");
    desc.add("calibSource", edm::ESInputTag{})->setComment("Label for calibration parameters");
    descriptions.addWithDefaultLabel(desc);
  }

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define this as a plug-in
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestHGCalRecHitESProducers);
