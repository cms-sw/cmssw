// Conditions of the z-score CM DNN, built once per IOV and copied to the device by the framework:
//   HGCalCMMLModuleParamHost: per module first digi, eRx bitmask, one-hot index, scope, cell area fractions
//   HGCalCMMLNormParamHost:   z-score input mean and 1/std, per-channel target mean and std

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

#include "CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalCMMLParamHost.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCMMLParamDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace F = hgcalcmml::Feature;

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

    nlohmann::json readJson(edm::FileInPath const& file) {
      std::ifstream stream(file.fullPath());
      return nlohmann::json::parse(stream);
    }

    std::vector<float> readFloats(nlohmann::json const& j, char const* key, size_t size, edm::FileInPath const& file) {
      auto v = j.at(key).get<std::vector<float>>();
      if (v.size() != size)
        throw cms::Exception("Configuration")
            << file.fullPath() << ": '" << key << "' needs " << size << " values, has " << v.size();
      return v;
    }
  }  // namespace

  class HGCalCMMLParamESProducer : public ESProducer {
  public:
    explicit HGCalCMMLParamESProducer(edm::ParameterSet const& iConfig);
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    std::optional<hgcalcmml::HGCalCMMLModuleParamHost> produceModules(HGCalDenseIndexInfoRcd const& iRecord);
    std::optional<hgcalcmml::HGCalCMMLNormParamHost> produceNorm(HGCalDenseIndexInfoRcd const& iRecord);

  private:
    edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> moduleIndexerToken_;
    std::map<std::string, int> moduleVocabulary_;          // "ML_F3WC_IH0180" -> one-hot index
    std::map<std::string, std::vector<float>> sfByShape_;  // cellareas.json SF by module shape ("ML_F")
    std::vector<float> inMean_, inInvStd_;                 // F::Size
    std::vector<float> tgtMean_, tgtStd_;                  // F::NCH, by within-module channel
  };

  HGCalCMMLParamESProducer::HGCalCMMLParamESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
    // manifest: TensorCollection feeds the SoA column as-is, so the feature order must match the kernels
    auto const manifest = iConfig.getParameter<edm::FileInPath>("modelManifest");
    auto const man = readJson(manifest);
    if (man.at("feature_names").get<std::vector<std::string>>() != expectedFeatureNames())
      throw cms::Exception("Configuration")
          << manifest.fullPath() << ": feature_names differ from the CM input kernels";
    auto const vocabulary = man.at("module_vocabulary").get<std::vector<std::string>>();
    for (size_t j = 0; j < vocabulary.size(); ++j)
      moduleVocabulary_[vocabulary[j]] = int(j);

    // z-score normalisation of the inputs and of the per-channel targets
    auto const preprocessing = iConfig.getParameter<edm::FileInPath>("inputPreprocessing");
    auto const pre = readJson(preprocessing);
    inMean_ = readFloats(pre, "mean", F::Size, preprocessing);
    inInvStd_ = readFloats(pre, "std", F::Size, preprocessing);
    for (auto& s : inInvStd_) {
      if (!(s > 0.f))
        throw cms::Exception("Configuration") << preprocessing.fullPath() << ": std must be > 0";
      s = 1.f / s;
    }
    tgtMean_ = readFloats(pre, "target_mean", F::NCH, preprocessing);
    tgtStd_ = readFloats(pre, "target_std", F::NCH, preprocessing);

    auto const cellAreas = iConfig.getParameter<edm::FileInPath>("cellAreas");
    auto const areas = readJson(cellAreas);
    for (auto const& [shape, table] : areas.items())
      sfByShape_[shape] = table.at("SF").get<std::vector<float>>();
    if (auto it = sfByShape_.find("ML_F"); it == sfByShape_.end() || it->second.size() != F::NCH)
      throw cms::Exception("Configuration") << cellAreas.fullPath() << ": ML_F needs " << F::NCH << " SF entries";

    auto cc = setWhatProduced(this, &HGCalCMMLParamESProducer::produceModules);
    moduleIndexerToken_ = cc.consumesFrom<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd>(
        iConfig.getParameter<edm::ESInputTag>("moduleIndexerSource"));
    setWhatProduced(this, &HGCalCMMLParamESProducer::produceNorm);
  }

  std::optional<hgcalcmml::HGCalCMMLModuleParamHost> HGCalCMMLParamESProducer::produceModules(
      HGCalDenseIndexInfoRcd const& iRecord) {
    auto const& moduleIndexer = iRecord.get(moduleIndexerToken_);
    uint32_t const nmodules = moduleIndexer.maxModulesCount();
    hgcalcmml::HGCalCMMLModuleParamHost product(cms::alpakatools::host(), nmodules);
    product.zeroInitialise();  // offsets, eRx, inScope and cellFrac default to 0
    auto view = product.view();
    for (uint32_t m = 0; m < nmodules; ++m)
      view[m].onehotIdx() = -1;

    for (auto const& fed : moduleIndexer.fedReadoutSequences()) {
      for (uint32_t modid = 0; modid < fed.chDataOffsets_.size(); ++modid) {
        uint32_t const m = moduleIndexer.getIndexForModule(fed.id, modid);
        if (m < nmodules) {
          view[m].chDataOffset() = fed.chDataOffsets_[modid];
          view[m].enabledErx() = fed.enabledErx_[modid];
        }
      }
    }
    // in scope: known shape ("ML-F3WC-IH0182" -> "ML_F"), NCH channels within the digi collection
    for (auto const& [typecode, ids] : moduleIndexer.typecodeMap()) {
      uint32_t const m = moduleIndexer.getIndexForModule(typecode);
      if (m >= nmodules)
        continue;
      auto row = view[m];
      std::string key = typecode;
      std::replace(key.begin(), key.end(), '-', '_');
      if (auto it = moduleVocabulary_.find(key); it != moduleVocabulary_.end())
        row.onehotIdx() = it->second;
      auto const sf = sfByShape_.find(key.substr(0, 4));
      row.inScope() = sf != sfByShape_.end() && sf->second.size() == F::NCH &&
                      uint32_t(__builtin_popcount(row.enabledErx())) * 37u == F::NCH &&
                      row.chDataOffset() + F::NCH <= moduleIndexer.maxDataSize();
      if (row.inScope())
        for (uint32_t ch = 0; ch < F::NCH; ++ch)
          row.cellFrac()(ch) = sf->second[ch];
    }
    return product;
  }

  std::optional<hgcalcmml::HGCalCMMLNormParamHost> HGCalCMMLParamESProducer::produceNorm(HGCalDenseIndexInfoRcd const&) {
    hgcalcmml::HGCalCMMLNormParamHost product(cms::alpakatools::host(), F::Size);
    auto view = product.view();
    for (uint32_t k = 0; k < F::Size; ++k) {
      view[k].inMean() = inMean_[k];
      view[k].inInvStd() = inInvStd_[k];
      view[k].tgtMean() = k < F::NCH ? tgtMean_[k] : 0.f;
      view[k].tgtStd() = k < F::NCH ? tgtStd_[k] : 0.f;
    }
    return product;
  }

  void HGCalCMMLParamESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    std::string const dir = "RecoLocalCalo/HGCalRecProducers/data/CMCalibrations/";
    edm::ParameterSetDescription desc;
    desc.add<edm::FileInPath>("modelManifest", edm::FileInPath(dir + "cm_dnn_multimodule_zscore_manifest.json"))
        ->setComment("feature order, module vocabulary");
    desc.add<edm::FileInPath>("inputPreprocessing",
                              edm::FileInPath(dir + "cm_dnn_multimodule_zscore_preprocessing.json"))
        ->setComment("z-score normalisation of the inputs and targets");
    desc.add<edm::FileInPath>("cellAreas", edm::FileInPath(dir + "cellareas.json"))
        ->setComment("cellareas.json (ML_F/MH_F SF)");
    desc.add<edm::ESInputTag>("moduleIndexerSource", edm::ESInputTag{})
        ->setComment("Label for HGCalMappingModuleIndexer");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(HGCalCMMLParamESProducer);
