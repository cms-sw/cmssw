// Author: Izaak Neutelings (March 2024)
// Based on: https://github.com/CMS-HGCAL/cmssw/blob/hgcal-condformat-HGCalNANO-13_2_0_pre3_linearity/RecoLocalCalo/HGCalRecAlgos/plugins/alpaka/HGCalRecHitProducer.cc
// CMSSW includes
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
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalCalibrationParameterHostCollection.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCalibrationParameterDeviceCollection.h"

template<class T> double duration(T t0,T t1) {
  auto elapsed_secs = t1-t0;
  typedef std::chrono::duration<float> float_seconds;
  auto secs = std::chrono::duration_cast<float_seconds>(elapsed_secs);
  return secs.count();
}

inline std::chrono::time_point<std::chrono::steady_clock> now() {
  return std::chrono::steady_clock::now();
}

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class TestHGCalRecHitESProducer : public stream::EDProducer<> {
  public:
    explicit TestHGCalRecHitESProducer(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produce(device::Event&, device::EventSetup const&) override;
    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    edm::ESWatcher<HGCalMappingModuleIndexerRcd> calibWatcher_;
    //edm::ESWatcher<HGCalCondSerializableConfigRcd> configWatcher_;
    device::ESGetToken<hgcalrechit::HGCalCalibParamDeviceCollection, HGCalMappingModuleIndexerRcd> calibToken_;
    //device::ESGetToken<hgcalrechit::HGCalConfigParamDeviceCollection, HGCalCondSerializableConfigRcd> configToken_;
  };

  TestHGCalRecHitESProducer::TestHGCalRecHitESProducer(const edm::ParameterSet& iConfig) {
    std::cout << "TestHGCalRecHitESProducer::TestHGCalRecHitESProducer" << std::endl;
    calibToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("calibSource"));
    //configToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("configSource"));
  }

  void TestHGCalRecHitESProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup){
    std::cout << "TestHGCalRecHitESProducer::beginRun" << std::endl;
  }

  void TestHGCalRecHitESProducer::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    std::cout << "TestHGCalRecHitESProducer::produce" << std::endl;
    auto queue = iEvent.queue();
    auto const& deviceCalibParamProvider = iSetup.getData(calibToken_);
    //auto const& deviceConfigParamProvider = iSetup.getData(configToken_);

    //// Check if there are new conditions and read them
    //if (configWatcher_.check(iSetup)){
    //  for(int i=0; i<deviceConfigParamProvider.view().metadata().size(); i++) {
    //      LogDebug("HGCalCalibrationParameter")
    //        << "gain = " << deviceConfigParamProvider.view()[i].gain();
    //  }
    //}

    // Check if there are new conditions and read them
    if(calibWatcher_.check(iSetup)){
      std::cout << "TestHGCalRecHitESProducer::produce: moduleMap.getMaxDataSize()=" << deviceCalibParamProvider.view().map().getMaxDataSize()
                << ", moduleMap.getMaxERxSize()=" << deviceCalibParamProvider.view().map().getMaxERxSize() << std::endl;
      std::cout << "TestHGCalRecHitESProducer::produce: device size=" << deviceCalibParamProvider.view().metadata().size() << std::endl;
      int size = std::min(250,deviceCalibParamProvider.view().metadata().size());
      std::cout << "  idx  hex  ADC_ped  CM_slope  CM_ped  BXm1_slope" << std::endl;
      for(int idx=0; idx<size; idx++) {
        std::cout << std::setw(5) << idx << std::hex << std::setw(5) << idx << std::dec
          << std::setw(9)  << deviceCalibParamProvider.view()[idx].ADC_ped()
          << std::setw(10) << deviceCalibParamProvider.view()[idx].CM_slope()
          << std::setw(8)  << deviceCalibParamProvider.view()[idx].CM_ped()
          << std::setw(12) << deviceCalibParamProvider.view()[idx].BXm1_slope()
          // << std::setw(6) << deviceCalibParamProvider.view()[idx].BXm1_offset() // redundant
          << std::endl;
      }
    }
  }

  void TestHGCalRecHitESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add("calibSource", edm::ESInputTag{})->setComment("Label for calibration parameters");
    //desc.add("configSource", edm::ESInputTag{})->setComment("Label for ROC configuration parameters");
    descriptions.addWithDefaultLabel(desc);
  }

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define this as a plug-in
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestHGCalRecHitESProducer);
