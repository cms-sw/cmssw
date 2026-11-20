// CMSSW imports
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include <iomanip>  // for std::setw
#include <future>
#include <chrono>
#include <optional>
#include <string>

// Alpaka imports
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingFixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// includes for data formats
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDevice.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"

// includes for size, calibration, and configuration parameters
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalRecHitCalibrationAlgorithms.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"
#include "CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h"

// includes for the common-mode DNN
#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalSoACMML.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCMCalibrationAlgorithms.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCMMLParamDevice.h"

// flag to assist the computational performance test
// #define HGCAL_PERF_TEST

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  // acquire: copy digis and calibration to the device, [useCMML] CM DNN inputs + inference, calibration
  // produce: zero-suppression selection
  // Fixed queue per stream for the PyTorch device caching allocator
  class HGCalRecHitsProducer : public stream::SynchronizingFixedQueueEDProducer<> {
  public:
    explicit HGCalRecHitsProducer(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void acquire(device::Event const&, device::EventSetup const&) override;
    void produce(device::Event&, device::EventSetup const&) override;

    // Fills deviceCorrections with the DNN CM correction in pedestal-subtracted ADC
    void runCMModel(device::Event const& iEvent,
                    device::EventSetup const& iSetup,
                    hgcaldigi::HGCalDigiDevice const& deviceDigis,
                    HGCalCalibParamDevice const& deviceCalib,
                    HGCalCMCorrectionDeviceCollection& deviceCorrections);

    edm::ESWatcher<HGCalElectronicsMappingRcd> calibWatcher_;
    const edm::EDGetTokenT<hgcaldigi::HGCalDigiHost> digisToken_;
    const edm::ESGetToken<hgcalrechit::HGCalCalibParamHost, HGCalModuleConfigurationRcd> calibToken_;
    const device::ESGetToken<hgcal::HGCalMappingCellParamDevice, HGCalElectronicsMappingRcd> mappingToken_;
    const device::ESGetToken<hgcal::HGCalDenseIndexInfoDevice, HGCalDenseIndexInfoRcd> indexingToken_;
    const device::ESGetToken<hgcal::HGCalMappingModuleParamDevice, HGCalElectronicsMappingRcd> moduleToken_;
    const device::EDPutToken<HGCalSoARecHitsDeviceCollection> recHitsToken_;
    const HGCalRecHitCalibrationAlgorithms calibrator_;
    const double k_noise_;
    const int n_hits_scale_;
    int ndigis_;
    host_buffer<int32_t> nsel_;
    std::optional<device_buffer<Device, int32_t[]>> sidx_;
    std::optional<HGCalSoARecHitsDeviceCollection> recHits_;

    // CM DNN
    const bool useCMML_;
    const bool cmAltAsNominal_;
    const HGCalCMCalibrationAlgorithms cmAlgo_;
    std::optional<torch::AlpakaModel> model_;
    // per-module parameters and z-score constants, built once per IOV by HGCalCMMLParamESProducer
    device::ESGetToken<HGCalCMMLModuleParamDevice, HGCalDenseIndexInfoRcd> cmModuleParamToken_;
    device::ESGetToken<HGCalCMMLNormParamDevice, HGCalDenseIndexInfoRcd> cmNormParamToken_;
  };

  HGCalRecHitsProducer::HGCalRecHitsProducer(const edm::ParameterSet& iConfig)
      : SynchronizingFixedQueueEDProducer(iConfig),
        digisToken_{consumes<hgcaldigi::HGCalDigiHost>(iConfig.getParameter<edm::InputTag>("digis"))},
        calibToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("calibSource"))},
        mappingToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("mappingSource"))},
        indexingToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("indexingSource"))},
        moduleToken_{esConsumes()},
        recHitsToken_{produces()},
        calibrator_{iConfig.getParameter<int>("n_blocks"), iConfig.getParameter<int>("n_threads")},
        k_noise_{iConfig.getParameter<double>("k_noise")},
        n_hits_scale_{iConfig.getParameter<int>("n_hits_scale")},
        nsel_{make_host_buffer<int32_t, Platform>()},
        useCMML_{iConfig.getParameter<bool>("useCMML")},
        cmAltAsNominal_{iConfig.getParameter<bool>("cmAltAsNominal")},
        cmAlgo_{iConfig.getParameter<int>("cm_n_threads")} {
#ifndef HGCAL_PERF_TEST
    if (n_hits_scale_ > 1) {
      throw cms::Exception("RuntimeError") << "Build with `HGCAL_PERF_TEST` flag to activate `n_hits_scale`.";
    }
#endif
    if (cmAltAsNominal_ && !useCMML_)
      throw cms::Exception("Configuration") << "cmAltAsNominal=True requires useCMML=True";
    if (!useCMML_)
      return;

    model_.emplace(edm::FileInPath(iConfig.getParameter<std::string>("model")).fullPath());
    auto const cmParamSource = iConfig.getParameter<edm::ESInputTag>("cmParamSource");
    cmModuleParamToken_ = esConsumes(cmParamSource);
    cmNormParamToken_ = esConsumes(cmParamSource);
  }

  void HGCalRecHitsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("digis", edm::InputTag("hgcalDigis", "DIGI", "TEST"));
    desc.add("calibSource", edm::ESInputTag{})->setComment("Label for calibration parameters");
    desc.add("mappingSource", edm::ESInputTag{})->setComment("Label for cell mapping parameters");
    desc.add("indexingSource", edm::ESInputTag{})->setComment("Label for cell dense indexer");
    desc.add<double>("k_noise", -100.)->setComment("ZS threshold for rechits (multiples of noise)");
    desc.add<int>("n_blocks", -1);
    desc.add<int>("n_threads", -1);
    desc.add<int>("n_hits_scale", -1);
    desc.add<bool>("useCMML", false)->setComment("Run the CM DNN for energy_variations(CM_ALT), else CM_ALT = energy");
    desc.add<bool>("cmAltAsNominal", false)
        ->setComment("use the DNN CM for energy (ZS, clustering), the linear CM for energy_variations(CM_ALT)");
    desc.add<std::string>("model", "RecoLocalCalo/HGCalRecProducers/data/CMCalibrations/cm_dnn_multimodule_zscore.pth")
        ->setComment("TorchScript z-score CM DNN");
    desc.add("cmParamSource", edm::ESInputTag{})->setComment("Label for HGCalCMMLParamESProducer");
    desc.add<int>("cm_n_threads", 256);
    descriptions.addWithDefaultLabel(desc);
  }

  void HGCalRecHitsProducer::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    auto& queue = iEvent.queue();

    // Read digis
    auto const& hostCalibParamProvider = iSetup.getData(calibToken_);
    auto const& deviceMappingCellParamProvider = iSetup.getData(mappingToken_);
    auto const& deviceIndexingParamProvider = iSetup.getData(indexingToken_);
    auto const& deviceModuleInfoProvider = iSetup.getData(moduleToken_);
    auto const& hostDigisIn = iEvent.get(digisToken_);

    //printout new conditions if available
    LogDebug("HGCalCalibrationParameter").log([&](auto& log) {
      if (calibWatcher_.check(iSetup)) {
        for (int i = 0; i < hostCalibParamProvider.view().metadata().size(); i++) {
          log << "idx = " << i << ", "
              << "ADC_ped = " << hostCalibParamProvider.view()[i].ADC_ped() << ", "
              << "CM_slope = " << hostCalibParamProvider.view()[i].CM_slope() << ", "
              << "CM_ped = " << hostCalibParamProvider.view()[i].CM_ped() << ", "
              << "BXm1_slope = " << hostCalibParamProvider.view()[i].BXm1_slope() << ", ";
        }
      }
    });

#ifdef HGCAL_PERF_TEST
    uint32_t oldSize = hostDigisIn.view().metadata().size();
    uint32_t newSize = oldSize * (n_hits_scale_ > 0 ? (unsigned)n_hits_scale_ : 1);
    auto hostDigis = HGCalDigiHost(newSize, queue);
    auto hostCalibParam = HGCalCalibParamHost(newSize, queue);
    // TODO: replace with memcp ?
    for (uint32_t i = 0; i < newSize; i++) {
      hostDigis.view()[i].tctp() = hostDigisIn.view()[i % oldSize].tctp();
      hostDigis.view()[i].adcm1() = hostDigisIn.view()[i % oldSize].adcm1();
      hostDigis.view()[i].adc() = hostDigisIn.view()[i % oldSize].adc();
      hostDigis.view()[i].tot() = hostDigisIn.view()[i % oldSize].tot();
      hostDigis.view()[i].toa() = hostDigisIn.view()[i % oldSize].toa();
      hostDigis.view()[i].cm() = hostDigisIn.view()[i % oldSize].cm();
      hostDigis.view()[i].flags() = hostDigisIn.view()[i % oldSize].flags();

      hostCalibParam.view()[i].ADC_ped() = hostCalibParamProvider.view()[i % oldSize].ADC_ped();
      hostCalibParam.view()[i].Noise() = hostCalibParamProvider.view()[i % oldSize].Noise();
      hostCalibParam.view()[i].CM_slope() = hostCalibParamProvider.view()[i % oldSize].CM_slope();
      hostCalibParam.view()[i].CM_ped() = hostCalibParamProvider.view()[i % oldSize].CM_ped();
      hostCalibParam.view()[i].BXm1_slope() = hostCalibParamProvider.view()[i % oldSize].BXm1_slope();
      hostCalibParam.view()[i].TOTtoADC() = hostCalibParamProvider.view()[i % oldSize].TOTtoADC();
      hostCalibParam.view()[i].TOT_ped() = hostCalibParamProvider.view()[i % oldSize].TOT_ped();
      hostCalibParam.view()[i].TOT_lin() = hostCalibParamProvider.view()[i % oldSize].TOT_lin();
      hostCalibParam.view()[i].TOT_P0() = hostCalibParamProvider.view()[i % oldSize].TOT_P0();
      hostCalibParam.view()[i].TOT_P1() = hostCalibParamProvider.view()[i % oldSize].TOT_P1();
      hostCalibParam.view()[i].TOT_P2() = hostCalibParamProvider.view()[i % oldSize].TOT_P2();
      hostCalibParam.view()[i].TOAtops() = hostCalibParamProvider.view()[i % oldSize].TOAtops();
      hostCalibParam.view()[i].MIPS_scale() = hostCalibParamProvider.view()[i % oldSize].MIPS_scale();
      hostCalibParam.view()[i].valid() = hostCalibParamProvider.view()[i % oldSize].valid();
    }
#else
    const auto& hostDigis = hostDigisIn;
    const auto& hostCalibParam = hostCalibParamProvider;
#endif

    LogDebug("HGCalRecHitsProducer") << "Loaded host digis: " << hostDigis.view().metadata().size();  //<< std::endl;

#ifdef EDM_ML_DEBUG
    alpaka::wait(queue);
    auto start = std::chrono::steady_clock::now();
#endif

    LogDebug("HGCalRecHitsProducer") << "\n\nINFO -- Copying the digis and calib to the device\n\n" << std::endl;
    ndigis_ = hostDigis.view().metadata().size();
    hgcaldigi::HGCalDigiDevice deviceDigis(queue, ndigis_);
    alpaka::memcpy(queue, deviceDigis.buffer(), hostDigis.const_buffer());
    HGCalCalibParamDevice deviceCalibParam(queue, hostCalibParam.view().metadata().size());
    alpaka::memcpy(queue, deviceCalibParam.buffer(), hostCalibParam.const_buffer());

    HGCalCMCorrectionDeviceCollection deviceCorrections(queue, ndigis_);
    if (useCMML_ && ndigis_ > 0)
      runCMModel(iEvent, iSetup, deviceDigis, deviceCalibParam, deviceCorrections);
    else
      deviceCorrections.zeroInitialise(queue);  // valid = 0: CM_ALT = nominal

#ifdef HGCAL_PERF_TEST
    auto tmpRecHits = calibrator_.calibrate(queue, hostDigis, deviceCalibParam);
    recHits_ = HGCalSoARecHitsDeviceCollection(oldSize, queue);
    alpaka::memcpy(queue, recHits.buffer(), tmpRecHits.const_buffer(), oldSize);
#else
    LogDebug("HGCalRecHitsProducer") << "\n\nINFO -- calling calibrate method";  //<< std::endl;
    *nsel_ = 0;
    sidx_ = make_device_buffer<int32_t[]>(queue, ndigis_);
    recHits_ = calibrator_.calibrate(queue,
                                     nsel_.data(),
                                     sidx_->data(),
                                     deviceDigis,
                                     deviceCalibParam,
                                     deviceModuleInfoProvider,
                                     deviceMappingCellParamProvider,
                                     deviceIndexingParamProvider,
                                     deviceCorrections,
                                     cmAltAsNominal_,
                                     k_noise_);
#endif

#ifdef EDM_ML_DEBUG
    alpaka::wait(queue);
    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<float> elapsed = stop - start;
    LogDebug("HGCalRecHitsProducer") << "Time spent calibrating " << hostDigis.view().metadata().size()
                                     << " digis: " << elapsed.count();  //<< std::endl;
#endif
  }

  void HGCalRecHitsProducer::runCMModel(device::Event const& iEvent,
                                        device::EventSetup const& iSetup,
                                        hgcaldigi::HGCalDigiDevice const& deviceDigis,
                                        HGCalCalibParamDevice const& deviceCalib,
                                        HGCalCMCorrectionDeviceCollection& deviceCorrections) {
    auto& queue = iEvent.queue();
    auto const& deviceIndex = iSetup.getData(indexingToken_);
    auto const& moduleParam = iSetup.getData(cmModuleParamToken_);
    auto const& norm = iSetup.getData(cmNormParamToken_);
    const uint32_t ndigis = uint32_t(ndigis_);
    const uint32_t nmodules = moduleParam.const_view().metadata().size();

    const uint32_t ncalib = uint32_t(deviceCalib.view().metadata().size());
    if (ncalib < ndigis)
      throw cms::Exception("LogicError") << ncalib << " calibration channels for " << ndigis << " digis";

    HGCalSoACMMLDeviceCollection deviceModuleFeatures(queue, nmodules);
    HGCalSoACMMLDeviceCollection deviceFeatures(queue, ndigis);
    cmAlgo_.fillModuleInputs(queue, ndigis, deviceDigis, moduleParam, deviceCalib, norm, deviceModuleFeatures);
    cmAlgo_.fillDigiInputs(
        queue, ndigis, deviceIndex, deviceModuleFeatures, moduleParam, norm, deviceFeatures, deviceCorrections);

    cms::torch::alpakatools::TensorCollection<Queue> inputs;
    inputs.template add<hgcalcmml::HGCalCMMLSoA>("features", deviceFeatures.const_view().records().features());
    cms::torch::alpakatools::TensorCollection<Queue> outputs;
    outputs.template add<hgcalcmml::HGCalCMCorrectionSoA>("correction",
                                                          deviceCorrections.view().records().correction());
    model_->forward(queue, inputs, outputs);
    cmAlgo_.denormalizeOutputs(queue, ndigis, deviceIndex, norm, deviceCorrections);
  }

  void HGCalRecHitsProducer::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    auto recHits = calibrator_.select(iEvent.queue(), ndigis_, nsel_.data(), sidx_->data(), *recHits_);
    sidx_.reset();

    LogDebug("HGCalRecHitsProducer") << "\n\nINFO -- storing rec hits in the event";  //<< std::endl;
    iEvent.emplace(recHitsToken_, std::move(recHits));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define this as a plug-in
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalRecHitsProducer);
