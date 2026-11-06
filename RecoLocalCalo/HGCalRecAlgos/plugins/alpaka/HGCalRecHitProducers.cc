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
#include <algorithm>
#include <atomic>
#include <iomanip>  // for std::setw
#include <future>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// Alpaka imports
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
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

// flag to assist the computational performance test
// #define HGCAL_PERF_TEST

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  namespace F = hgcalcmml::Feature;

  // acquire: copy digis and calibration to the device, [useCMML] CM DNN inputs + inference, calibration
  // produce: zero-suppression selection
  class HGCalRecHitsProducer : public stream::SynchronizingEDProducer<> {
  public:
    explicit HGCalRecHitsProducer(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void acquire(device::Event const&, device::EventSetup const&) override;
    void produce(device::Event&, device::EventSetup const&) override;

    void loadCMModelMetadata(std::string const& manifestPath, std::string const& preprocessingPath);
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
    const edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> moduleIndexerToken_;
    const device::EDPutToken<HGCalSoARecHitsDeviceCollection> recHitsToken_;
    const HGCalRecHitCalibrationAlgorithms calibrator_;
    const double k_noise_;
    const int n_hits_scale_;
    int ndigis_;
    cms::alpakatools::host_buffer<int32_t> nsel_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> sidx_;
    std::optional<HGCalSoARecHitsDeviceCollection> recHits_;

    // CM DNN
    const bool useCMML_;
    const bool cmAltAsNominal_;
    const HGCalCMCalibrationAlgorithms cmAlgo_;
    std::optional<torch::AlpakaModel> model_;
    std::vector<std::string> featureNames_;
    std::map<std::string, int> moduleVocabulary_;  // "ML_F3WC_IH0180" -> one-hot index
    bool normalize_ = false;                       // z-score model
    std::vector<float> inMean_, inInvStd_;         // F::Size
    std::vector<float> tgtMean_, tgtStd_;          // F::NCH, by within-module channel
    // cellareas.json cell area fractions by module shape ("ML_F"), by within-module channel
    std::map<std::string, std::vector<float>> h_sfByShape_;

    // compareCellAreas: print cellareas.json vs HGCalDenseIndexInfo area() for every channel of full wafers, once per job
    const bool compareCellAreas_;
    edm::ESGetToken<hgcal::HGCalDenseIndexInfoHost, HGCalDenseIndexInfoRcd> hostIndexingToken_;
    std::map<std::string, std::vector<float>> h_areaByShape_;  // cellareas.json Area [mm2], same indexing
    void printCellAreaComparison(device::EventSetup const& iSetup,
                                 std::vector<std::string> const& typecodes,
                                 uint32_t nmodules) const;
  };

  namespace {
    // feature names in the order the kernels fill them
    std::vector<std::string> expectedFeatureNames() {
      std::vector<std::string> names;
      char buf[32];
      auto add = [&](char const* fmt, unsigned i) {
        std::snprintf(buf, sizeof(buf), fmt, i);
        names.emplace_back(buf);
      };
      for (uint32_t e = 0; e < F::NCM; ++e)
        add("cm_erx%02u_pedsub", e);
      names.insert(names.end(), {"nchtoa", "nchtot"});
      for (float t : hgcalcmml::kNchAdcGtThresholds)
        add("nchadcgt%u", unsigned(t));
      for (uint32_t j = 0; j < F::NONEHOT; ++j)
        add("module_onehot_%03u", j);
      for (uint32_t i = 0; i < F::NCH; ++i)
        add("adc_allch_%03u", i);
      names.insert(names.end(), {"channel_indices", "erx_indices", "cell_area_fraction"});
      return names;
    }
    // Json helpers for the manifest, the z-score mean/std file and the cell areas
    // I have not bothered to code review these helper functions and have also not looked into if better way to do this
    nlohmann::json readJson(std::string const& path) {
      std::ifstream file(path);
      if (!file.is_open())
        throw cms::Exception("Configuration") << "Cannot open " << path;
      return nlohmann::json::parse(file);
    }

    // cellareas.json key of a typecode: "ML-F3WC-IH0182" -> "ML_F"
    std::string moduleShape(std::string const& typecode) {
      std::string shape = typecode.substr(0, 4);
      std::replace(shape.begin(), shape.end(), '-', '_');
      return shape;
    }

    std::vector<float> readFloats(nlohmann::json const& j, char const* key, size_t size, std::string const& file) {
      auto v = j.at(key).get<std::vector<float>>();
      if (v.size() != size || !std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); }))
        throw cms::Exception("Configuration")
            << file << ": '" << key << "' needs " << size << " finite values, has " << v.size();
      return v;
    }

    template <typename T>
    device_buffer<Device, T[]> toDevice(Queue& queue, std::vector<T> const& v) {
      auto buf = make_device_buffer<T[]>(queue, v.size());
      alpaka::memcpy(queue, buf, make_host_view(v.data(), v.size()));
      return buf;
    }
  }  // namespace

  HGCalRecHitsProducer::HGCalRecHitsProducer(const edm::ParameterSet& iConfig)
      : SynchronizingEDProducer(iConfig),
        digisToken_{consumes<hgcaldigi::HGCalDigiHost>(iConfig.getParameter<edm::InputTag>("digis"))},
        calibToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("calibSource"))},
        mappingToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("mappingSource"))},
        indexingToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("indexingSource"))},
        moduleToken_{esConsumes()},
        moduleIndexerToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("moduleIndexerSource"))},
        recHitsToken_{produces()},
        calibrator_{iConfig.getParameter<int>("n_blocks"), iConfig.getParameter<int>("n_threads")},
        k_noise_{iConfig.getParameter<double>("k_noise")},
        n_hits_scale_{iConfig.getParameter<int>("n_hits_scale")},
        nsel_{cms::alpakatools::make_host_buffer<int32_t, Platform>()},
        useCMML_{iConfig.getParameter<bool>("useCMML")},
        cmAltAsNominal_{iConfig.getParameter<bool>("cmAltAsNominal")},
        cmAlgo_{iConfig.getParameter<int>("cm_n_threads")},
        compareCellAreas_{iConfig.getParameter<bool>("compareCellAreas")} {
#ifndef HGCAL_PERF_TEST
    if (n_hits_scale_ > 1) {
      throw cms::Exception("RuntimeError") << "Build with `HGCAL_PERF_TEST` flag to activate `n_hits_scale`.";
    }
#endif
    if (cmAltAsNominal_ && !useCMML_)
      throw cms::Exception("Configuration") << "cmAltAsNominal=True requires useCMML=True";
    if (!useCMML_)
      return;

    auto const model = iConfig.getParameter<std::string>("model");
    auto const manifest = iConfig.getParameter<std::string>("modelManifest");
    auto const preprocessing = iConfig.getParameter<std::string>("inputPreprocessing");
    auto const cellAreas = iConfig.getParameter<std::string>("cellAreas");
    if (model.empty() || manifest.empty() || cellAreas.empty())
      throw cms::Exception("Configuration") << "useCMML=True requires 'model', 'modelManifest' and 'cellAreas'";
    model_.emplace(edm::FileInPath(model).fullPath());
    loadCMModelMetadata(edm::FileInPath(manifest).fullPath(),
                        preprocessing.empty() ? std::string() : edm::FileInPath(preprocessing).fullPath());

    auto const areas = readJson(edm::FileInPath(cellAreas).fullPath());
    for (auto const& [shape, table] : areas.items())
      h_sfByShape_[shape] = table.at("SF").get<std::vector<float>>();
    if (auto it = h_sfByShape_.find("ML_F"); it == h_sfByShape_.end() || it->second.size() != F::NCH)
      throw cms::Exception("Configuration") << "cellAreas: ML_F needs " << F::NCH << " SF entries";
    if (compareCellAreas_) {
      hostIndexingToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("indexingSource"));
      for (auto const& [shape, table] : areas.items())
        h_areaByShape_[shape] = table.at("Area").get<std::vector<float>>();
    }
  }

  void HGCalRecHitsProducer::loadCMModelMetadata(std::string const& manifestPath,
                                                 std::string const& preprocessingPath) {
    auto const man = readJson(manifestPath);
    // A bunch of saftety checks claude has decided to write, I will leave them as they could be useful for now.
    // It is based on a json from Arne per model which encodes NN feature information to be very explicit.
    // Depending on the desired brevity to sefety this could be removed.
    // TensorCollection feeds the SoA column as-is, so the manifest order must match the kernels
    featureNames_ = man.at("feature_names").get<std::vector<std::string>>();
    if (featureNames_ != expectedFeatureNames())
      throw cms::Exception("Configuration") << manifestPath << ": feature_names differ from the CM input kernels";
    if (man.at("nch").get<uint32_t>() != F::NCH)
      throw cms::Exception("Configuration") << manifestPath << ": nch != " << F::NCH;

    auto const vocabulary = man.at("module_vocabulary").get<std::vector<std::string>>();
    if (vocabulary.size() + 1 != F::NONEHOT || man.at("unknown_module_index").get<size_t>() != vocabulary.size())
      throw cms::Exception("Configuration") << manifestPath << ": vocabulary + UNKNOWN != " << F::NONEHOT;
    for (size_t j = 0; j < vocabulary.size(); ++j)
      moduleVocabulary_[vocabulary[j]] = int(j);

    // if normalise json exists, we know we need to normalise the inputs!
    normalize_ = man.at("preprocess_inputs").get<bool>();
    if (normalize_ == preprocessingPath.empty())
      throw cms::Exception("Configuration")
          << manifestPath << ": preprocess_inputs=" << normalize_ << " requires 'inputPreprocessing' to be "
          << (normalize_ ? "set" : "empty");
    if (!normalize_)
      return;

    // Store all the noramlisation inputs.
    auto const pre = readJson(preprocessingPath);
    if (!pre.at("enabled").get<bool>() || pre.at("method") != "zscore" || !pre.at("targets_enabled").get<bool>() ||
        pre.at("target_method") != "per_channel_zscore")
      throw cms::Exception("Configuration")
          << preprocessingPath << ": expected zscore inputs, per_channel_zscore targets";
    if (pre.at("feature_names").get<std::vector<std::string>>() != featureNames_)
      throw cms::Exception("Configuration") << preprocessingPath << ": feature_names differ from the manifest";
    std::vector<uint32_t> channels(F::NCH);
    for (uint32_t i = 0; i < F::NCH; ++i)
      channels[i] = i;
    if (pre.at("target_channels").get<std::vector<uint32_t>>() != channels)
      throw cms::Exception("Configuration") << preprocessingPath << ": target_channels is not 0.." << F::NCH - 1;

    inMean_ = readFloats(pre, "mean", F::Size, preprocessingPath);
    inInvStd_ = readFloats(pre, "std", F::Size, preprocessingPath);
    for (auto& s : inInvStd_) {
      if (!(s > 0.f))
        throw cms::Exception("Configuration") << preprocessingPath << ": std must be > 0";
      s = 1.f / s;
    }
    tgtMean_ = readFloats(pre, "target_mean", F::NCH, preprocessingPath);
    tgtStd_ = readFloats(pre, "target_std", F::NCH, preprocessingPath);
  }

  // One line per channel of full Si wafers: cellareas.json (used by the DNN) vs HGCalDenseIndexInfo area(),
  // until the dense index area is validated and can replace the json. Printed once per job, not per stream.
  void HGCalRecHitsProducer::printCellAreaComparison(device::EventSetup const& iSetup,
                                                     std::vector<std::string> const& typecodes,
                                                     uint32_t nmodules) const {
    static std::atomic<bool> printed{false};
    if (printed.exchange(true))
      return;
    auto const& hostIndex = iSetup.getData(hostIndexingToken_);
    uint32_t ncompared = 0, ndiffer = 0;
    for (int32_t i = 0; i < hostIndex.view().metadata().size(); ++i) {
      auto const row = hostIndex.view()[i];
      uint32_t const m = row.modInfoIdx(), ch = row.chNumber();
      if (m >= nmodules || typecodes[m].rfind('M', 0) != 0 || typecodes[m].find("-F") == std::string::npos)
        continue;
      std::string const shape = moduleShape(typecodes[m]);
      if (!h_sfByShape_.count(shape) || !h_areaByShape_.count(shape))
        continue;
      auto const& sf = h_sfByShape_.at(shape);
      auto const& area = h_areaByShape_.at(shape);
      if (ch >= sf.size() || ch >= area.size())
        continue;
      bool const same = std::abs(row.area() * 100.f - area[ch]) < 0.01f;  // area() in cm2, json in mm2
      ++ncompared;
      ndiffer += !same;
      edm::LogPrint("HGCalCellAreaCheck")
          << "CELL AREA module=" << typecodes[m] << " ch=" << ch << " area_frac_json=" << sf[ch]
          << " area_json[mm2]=" << area[ch] << " area()[cm2]=" << row.area() << (same ? "" : " DIFFERS");
    }
    edm::LogPrint("HGCalCellAreaCheck") << "CELL AREA summary: " << ncompared << " channels compared, " << ndiffer
                                        << " differ";
  }

  void HGCalRecHitsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("digis", edm::InputTag("hgcalDigis", "DIGI", "TEST"));
    desc.add("calibSource", edm::ESInputTag{})->setComment("Label for calibration parameters");
    desc.add("mappingSource", edm::ESInputTag{})->setComment("Label for cell mapping parameters");
    desc.add("indexingSource", edm::ESInputTag{})->setComment("Label for cell dense indexer");
    desc.add("moduleIndexerSource", edm::ESInputTag{})->setComment("Label for HGCalMappingModuleIndexer");
    desc.add<double>("k_noise", -100.)->setComment("ZS threshold for rechits (multiples of noise)");
    desc.add<int>("n_blocks", -1);
    desc.add<int>("n_threads", -1);
    desc.add<int>("n_hits_scale", -1);
    desc.add<bool>("useCMML", false)->setComment("Run the CM DNN for energy_variations(CM_ALT), else CM_ALT = energy");
    desc.add<bool>("cmAltAsNominal", false)
        ->setComment("use the DNN CM for energy (ZS, clustering), the linear CM for energy_variations(CM_ALT)");
    desc.add<std::string>("model", "RecoLocalCalo/HGCalRecProducers/data/CMCalibrations/cm_dnn_multimodule_zscore.pth")
        ->setComment("TorchScript CM DNN");
    desc.add<std::string>("modelManifest",
                          "RecoLocalCalo/HGCalRecProducers/data/CMCalibrations/cm_dnn_multimodule_zscore_manifest.json")
        ->setComment("feature order, module vocabulary, preprocess_inputs");
    desc.add<std::string>(
            "inputPreprocessing",
            "RecoLocalCalo/HGCalRecProducers/data/CMCalibrations/cm_dnn_multimodule_zscore_preprocessing.json")
        ->setComment("z-score normalisation JSON (preprocess_inputs=true), empty otherwise");
    desc.add<std::string>("cellAreas", "RecoLocalCalo/HGCalRecProducers/data/CMCalibrations/cellareas.json")
        ->setComment("cellareas.json (ML_F/MH_F SF)");
    desc.add<int>("cm_n_threads", 256);
    desc.add<bool>("compareCellAreas", false)
        ->setComment("print cellareas.json vs dense index area() for every channel of full wafers, once per job");
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
    sidx_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, ndigis_);
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
    auto const& moduleIndexer = iSetup.getData(moduleIndexerToken_);
    auto const& deviceIndex = iSetup.getData(indexingToken_);
    const uint32_t ndigis = uint32_t(ndigis_);
    const uint32_t nmodules = moduleIndexer.maxModulesCount();

    // per module: first digi, eRx bitmask, typecode, one-hot index, in scope (NCH channels, known shape), area fractions
    std::vector<uint32_t> h_chDataOffsets(nmodules, 0u), h_enabledErx(nmodules, 0u);
    for (const auto& fed : moduleIndexer.fedReadoutSequences()) {
      for (uint32_t modid = 0; modid < fed.chDataOffsets_.size(); ++modid) {
        uint32_t m = moduleIndexer.getIndexForModule(fed.id, modid);
        if (m < nmodules) {
          h_chDataOffsets[m] = fed.chDataOffsets_[modid];
          h_enabledErx[m] = fed.enabledErx_[modid];
        }
      }
    }
    std::vector<std::string> h_typecode(nmodules);
    for (const auto& kv : moduleIndexer.typecodeMap()) {
      uint32_t m = moduleIndexer.getIndexForModule(kv.first);
      if (m < nmodules)
        h_typecode[m] = kv.first;
    }
    std::vector<int> h_onehotIdx(nmodules, -1);
    std::vector<unsigned char> h_inScope(nmodules, 0);
    std::vector<float> h_sf(nmodules * F::NCH, 0.f);  // [m * NCH + chIdx]
    for (uint32_t m = 0; m < nmodules; ++m) {
      std::string key = h_typecode[m];
      std::replace(key.begin(), key.end(), '-', '_');
      if (auto it = moduleVocabulary_.find(key); it != moduleVocabulary_.end())
        h_onehotIdx[m] = it->second;
      auto const sf = h_sfByShape_.find(moduleShape(h_typecode[m]));
      h_inScope[m] = sf != h_sfByShape_.end() && sf->second.size() == F::NCH &&
                     uint32_t(__builtin_popcount(h_enabledErx[m])) * 37u == F::NCH &&
                     h_chDataOffsets[m] + F::NCH <= ndigis;
      if (h_inScope[m])
        std::copy(sf->second.begin(), sf->second.end(), h_sf.begin() + m * F::NCH);
    }
    if (compareCellAreas_)
      printCellAreaComparison(iSetup, h_typecode, nmodules);

    const uint32_t ncalib = uint32_t(deviceCalib.view().metadata().size());
    if (ncalib < ndigis)
      throw cms::Exception("LogicError") << ncalib << " calibration channels for " << ndigis << " digis";

    auto d_chDataOffsets = toDevice(queue, h_chDataOffsets);
    auto d_enabledErx = toDevice(queue, h_enabledErx);
    auto d_onehotIdx = toDevice(queue, h_onehotIdx);
    auto d_inScope = toDevice(queue, h_inScope);
    auto d_sf = toDevice(queue, h_sf);
    std::optional<device_buffer<Device, float[]>> d_inMean, d_inInvStd, d_tgtMean, d_tgtStd;
    if (normalize_) {
      d_inMean = toDevice(queue, inMean_);
      d_inInvStd = toDevice(queue, inInvStd_);
      d_tgtMean = toDevice(queue, tgtMean_);
      d_tgtStd = toDevice(queue, tgtStd_);
    }
    float const* inMean = normalize_ ? d_inMean->data() : nullptr;
    float const* inInvStd = normalize_ ? d_inInvStd->data() : nullptr;

    HGCalSoACMMLDeviceCollection deviceModuleFeatures(queue, nmodules);
    HGCalSoACMMLDeviceCollection deviceFeatures(queue, ndigis);
    cmAlgo_.fillModuleInputs(queue,
                             nmodules,
                             ndigis,
                             deviceDigis,
                             d_chDataOffsets.data(),
                             d_enabledErx.data(),
                             d_onehotIdx.data(),
                             d_inScope.data(),
                             deviceCalib,
                             normalize_,
                             inMean,
                             inInvStd,
                             deviceModuleFeatures);
    cmAlgo_.fillDigiInputs(queue,
                           nmodules,
                           ndigis,
                           deviceIndex,
                           deviceModuleFeatures,
                           d_enabledErx.data(),
                           d_inScope.data(),
                           d_sf.data(),
                           normalize_,
                           inMean,
                           inInvStd,
                           deviceFeatures,
                           deviceCorrections);

    cms::torch::alpakatools::TensorCollection<Queue> inputs;
    inputs.template add<hgcalcmml::HGCalCMMLSoA>("features", deviceFeatures.const_view().records().features());
    cms::torch::alpakatools::TensorCollection<Queue> outputs;
    outputs.template add<hgcalcmml::HGCalCMCorrectionSoA>("correction",
                                                          deviceCorrections.view().records().correction());
    model_->forward(queue, inputs, outputs);

    if (normalize_)
      cmAlgo_.denormalizeOutputs(queue, ndigis, deviceIndex, d_tgtMean->data(), d_tgtStd->data(), deviceCorrections);
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
