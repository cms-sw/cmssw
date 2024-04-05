// CMSSW imports
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include <iomanip> // for std::setw
#include <future>
 
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
#include "CondFormats/DataRecord/interface/HGCalMappingModuleIndexerRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalRecHitCalibrationAlgorithms.h"

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

  class HGCalRecHitProducer : public stream::EDProducer<> {
  public:
    explicit HGCalRecHitProducer(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produce(device::Event&, device::EventSetup const&) override;
    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    edm::ESWatcher<HGCalMappingModuleIndexerRcd> calibWatcher_;
    edm::ESWatcher<HGCalModuleConfigurationRcd> configWatcher_;
    const edm::EDGetTokenT<hgcaldigi::HGCalDigiHost> digisToken_;
    device::ESGetToken<hgcalrechit::HGCalCalibParamDevice, HGCalMappingModuleIndexerRcd> calibToken_;
    device::ESGetToken<hgcalrechit::HGCalConfigParamDevice, HGCalModuleConfigurationRcd> configToken_;
    const device::EDPutToken<hgcalrechit::HGCalRecHitDevice> recHitsToken_;
    HGCalRecHitCalibrationAlgorithms calibrator_;  // cannot be "const" because the calibrate() method is not const
    int n_hits_scale;
  };

  HGCalRecHitProducer::HGCalRecHitProducer(const edm::ParameterSet& iConfig)
      : digisToken_{consumes<hgcaldigi::HGCalDigiHost>(iConfig.getParameter<edm::InputTag>("digis"))},
        recHitsToken_{produces()},
        calibrator_{HGCalRecHitCalibrationAlgorithms(
          iConfig.getParameter<int>("n_blocks"),
          iConfig.getParameter<int>("n_threads"))},
        n_hits_scale{iConfig.getParameter<int>("n_hits_scale")}
    {
      calibToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("calibSource"));
      configToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("configSource"));
    }

  void HGCalRecHitProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup){
    // auto moduleInfo = iSetup.getData(moduleInfoToken_);
    // std::tuple<uint16_t,uint8_t,uint8_t,uint8_t> denseIdxMax = moduleInfo.getMaxValuesForDenseIndex();    
    // calibrationParameterProvider_.initialize(HGCalCalibrationParameterProviderConfig{.EventSLinkMax=std::get<0>(denseIdxMax),
    //                                   .sLinkCaptureBlockMax=std::get<1>(denseIdxMax),
    //                                   .captureBlockECONDMax=std::get<2>(denseIdxMax),
    //                                   .econdERXMax=std::get<3>(denseIdxMax),
    //                                   .erxChannelMax = 37+2,//+2 for the two common modes
    // });
  }

  void HGCalRecHitProducer::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    auto queue = iEvent.queue();

    // Read digis
    auto const& deviceCalibParamProvider = iSetup.getData(calibToken_);
    auto const& deviceConfigParamProvider = iSetup.getData(configToken_);
    auto const& hostDigisIn = iEvent.get(digisToken_);

    // Check if there are new conditions and read them
    #ifdef EDM_ML_DEBUG
    if (calibWatcher_.check(iSetup)){
      for(int i=0; i<deviceConfigParamProvider.view().metadata().size(); i++) {
          LogDebug("HGCalCalibrationParameter")
            << "gain = " << deviceConfigParamProvider.view()[i].gain();
      }
    }
    #endif

    // Check if there are new conditions and read them
    #ifdef EDM_ML_DEBUG
    if (configWatcher_.check(iSetup)){
      for(int i=0; i<deviceCalibParamProvider.view().metadata().size(); i++) {
          LogDebug("HGCalCalibrationParameter")
              << "idx = "         << i << ", "
              << "ADC_ped = "     << deviceCalibParamProvider.view()[i].ADC_ped()    << ", "
              << "CM_slope = "    << deviceCalibParamProvider.view()[i].CM_slope()   << ", "
              << "CM_ped = "      << deviceCalibParamProvider.view()[i].CM_ped()     << ", "
              << "BXm1_slope = "  << deviceCalibParamProvider.view()[i].BXm1_slope() << ", "
              //<< "BXm1_offset = " << deviceCalibParamProvider.view()[i].BXm1_offset() // redundant
              << std::endl;
      }
    }
    #endif

    int oldSize = hostDigisIn.view().metadata().size();
    int newSize = oldSize * n_hits_scale;
    auto hostDigis = HGCalDigiHost(newSize, queue);
    // TODO: replace with memcp ?
    for(int i=0; i<newSize;i++){
      //hostDigis.view()[i].electronicsId() = hostDigisIn.view()[i%oldSize].electronicsId();
      hostDigis.view()[i].tctp()  = hostDigisIn.view()[i%oldSize].tctp();
      hostDigis.view()[i].adcm1() = hostDigisIn.view()[i%oldSize].adcm1();
      hostDigis.view()[i].adc()   = hostDigisIn.view()[i%oldSize].adc();
      hostDigis.view()[i].tot()   = hostDigisIn.view()[i%oldSize].tot();
      hostDigis.view()[i].toa()   = hostDigisIn.view()[i%oldSize].toa();
      hostDigis.view()[i].cm()    = hostDigisIn.view()[i%oldSize].cm();
      hostDigis.view()[i].flags() = hostDigisIn.view()[i%oldSize].flags();
      //LogDebug("HGCalCalibrationParameter")
      //  << "idx=" << i << ", elecId=" << hostDigis.view()[i].electronicsId()
      //  << ", cm=" << hostDigis.view()[i].cm() << std::endl;
    }
    LogDebug("HGCalRecHitProducer") << "Loaded host digis: " << hostDigis.view().metadata().size(); //<< std::endl;

    LogDebug("HGCalRecHitProducer") << "\n\nINFO -- calling calibrate method"; //<< std::endl;
    auto start = now();
    auto recHits = calibrator_.calibrate(queue, hostDigis, deviceCalibParamProvider, deviceConfigParamProvider);
    alpaka::wait(queue);
    auto stop = now();
    LogDebug("HGCalRecHitProducer") << "Time: " << duration(start, stop); //<< std::endl;

    LogDebug("HGCalRecHitProducer") << "\n\nINFO -- storing rec hits in the event"; //<< std::endl;
    iEvent.emplace(recHitsToken_, std::move(*recHits));
  }

  void HGCalRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("digis", edm::InputTag("hgcalDigis", "DIGI", "TEST"));
    desc.add("calibSource", edm::ESInputTag{})->setComment("Label for calibration parameters");
    desc.add("configSource", edm::ESInputTag{})->setComment("Label for ROC configuration parameters");
    desc.add<int>("n_blocks", -1);
    desc.add<int>("n_threads", -1);
    desc.add<int>("n_hits_scale", -1);
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define this as a plug-in
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalRecHitProducer);
