// CMSSW imports
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include <iomanip>  // for std::setw
#include <future>
#include <chrono>

// Alpaka imports
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

// includes for data formats
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDevice.h"
#include "DataFormats/HGCalRecHit/interface/HGCalRecHitHost.h"
#include "DataFormats/HGCalRecHit/interface/alpaka/HGCalRecHitDevice.h"

// includes for size, calibration, and configuration parameters
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalRecHitCalibrationAlgorithms.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class HGCalRecHitsProducer : public stream::EDProducer<> {
  public:
    explicit HGCalRecHitsProducer(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produce(device::Event&, device::EventSetup const&) override;
    edm::ESWatcher<HGCalElectronicsMappingRcd> calibWatcher_;
    edm::ESWatcher<HGCalModuleConfigurationRcd> configWatcher_;
    const edm::EDGetTokenT<hgcaldigi::HGCalDigiHost> digisToken_;
    device::ESGetToken<hgcalrechit::HGCalCalibParamDevice, HGCalModuleConfigurationRcd> calibToken_;
    device::ESGetToken<hgcalrechit::HGCalConfigParamDevice, HGCalModuleConfigurationRcd> configToken_;
    const device::EDPutToken<hgcalrechit::HGCalRecHitDevice> recHitsToken_;
    const HGCalRecHitCalibrationAlgorithms calibrator_;
    int n_hits_scale;
  };

  HGCalRecHitsProducer::HGCalRecHitsProducer(const edm::ParameterSet& iConfig)
      : digisToken_{consumes<hgcaldigi::HGCalDigiHost>(iConfig.getParameter<edm::InputTag>("digis"))},
        calibToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("calibSource"))},
        configToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("configSource"))},
        recHitsToken_{produces()},
        calibrator_{iConfig.getParameter<int>("n_blocks"), iConfig.getParameter<int>("n_threads")},
        n_hits_scale{iConfig.getParameter<int>("n_hits_scale")} {}

  void HGCalRecHitsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("digis", edm::InputTag("hgcalDigis", "DIGI", "TEST"));
    desc.add("calibSource", edm::ESInputTag{})->setComment("Label for calibration parameters");
    desc.add("configSource", edm::ESInputTag{})->setComment("Label for ROC configuration parameters");
    desc.add<int>("n_blocks", -1);
    desc.add<int>("n_threads", -1);
    desc.add<int>("n_hits_scale", -1);
    descriptions.addWithDefaultLabel(desc);
  }

  void HGCalRecHitsProducer::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    auto& queue = iEvent.queue();

    // Read digis
    auto const& deviceCalibParamProvider = iSetup.getData(calibToken_);
    auto const& deviceConfigParamProvider = iSetup.getData(configToken_);
    auto const& hostDigisIn = iEvent.get(digisToken_);

    //printout new conditions if available
    LogDebug("HGCalCalibrationParameter").log([&](auto& log) {
      if (calibWatcher_.check(iSetup)) {
        for (int i = 0; i < deviceConfigParamProvider.view().metadata().size(); i++) {
          log << "idx = " << i << ", "
              << "gain = " << deviceConfigParamProvider.view()[i].gain() << ","
              << "ADC_ped = " << deviceCalibParamProvider.view()[i].ADC_ped() << ", "
              << "CM_slope = " << deviceCalibParamProvider.view()[i].CM_slope() << ", "
              << "CM_ped = " << deviceCalibParamProvider.view()[i].CM_ped() << ", "
              << "BXm1_slope = " << deviceCalibParamProvider.view()[i].BXm1_slope() << ", " << std::endl;
        }
      }
    });

    int oldSize = hostDigisIn.view().metadata().size();
    int newSize = oldSize * n_hits_scale;
    auto hostDigis = HGCalDigiHost(newSize, queue);
    // TODO: replace with memcp ?
    for (int i = 0; i < newSize; i++) {
      hostDigis.view()[i].tctp() = hostDigisIn.view()[i % oldSize].tctp();
      hostDigis.view()[i].adcm1() = hostDigisIn.view()[i % oldSize].adcm1();
      hostDigis.view()[i].adc() = hostDigisIn.view()[i % oldSize].adc();
      hostDigis.view()[i].tot() = hostDigisIn.view()[i % oldSize].tot();
      hostDigis.view()[i].toa() = hostDigisIn.view()[i % oldSize].toa();
      hostDigis.view()[i].cm() = hostDigisIn.view()[i % oldSize].cm();
      hostDigis.view()[i].flags() = hostDigisIn.view()[i % oldSize].flags();
    }
    LogDebug("HGCalRecHitsProducer") << "Loaded host digis: " << hostDigis.view().metadata().size();  //<< std::endl;

    LogDebug("HGCalRecHitsProducer") << "\n\nINFO -- calling calibrate method";  //<< std::endl;

#ifdef EDM_ML_DEBUG
    alpaka::wait(queue);
    auto start = std::chrono::high_resolution_clock::now();
#endif

    auto recHits = calibrator_.calibrate(queue, hostDigis, deviceCalibParamProvider, deviceConfigParamProvider);

#ifdef EDM_ML_DEBUG
    alpaka::wait(queue);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = stop - start;
    LogDebug("HGCalRecHitsProducer") << "Time spent calibrating: " << elapsed.count();  //<< std::endl;
#endif

    LogDebug("HGCalRecHitsProducer") << "\n\nINFO -- storing rec hits in the event";  //<< std::endl;
    iEvent.emplace(recHitsToken_, std::move(recHits));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define this as a plug-in
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalRecHitsProducer);
