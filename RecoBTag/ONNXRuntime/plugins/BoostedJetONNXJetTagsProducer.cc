#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/DeepBoostedJetTagInfo.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include "RecoBTag/FeatureTools/interface/deep_helpers.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <nlohmann/json.hpp>

using namespace cms::Ort;
using namespace btagbtvdeep;

class BoostedJetONNXJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit BoostedJetONNXJetTagsProducer(const edm::ParameterSet &, const ONNXRuntime *);
  ~BoostedJetONNXJetTagsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &);
  static void globalEndJob(const ONNXRuntime *);

private:
  typedef std::vector<reco::DeepBoostedJetTagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  void produce(edm::Event &, const edm::EventSetup &) override;

  void make_inputs(const reco::DeepBoostedJetTagInfo &taginfo);

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;               // names of the output scores
  edm::EDGetTokenT<edm::View<reco::Jet>> jet_token_;  // jets if function produces a ValueMap
  std::vector<std::string> input_names_;              // names of each input group - the ordering is important!
  std::vector<std::vector<int64_t>> input_shapes_;    // shapes of each input group (-1 for dynamic axis)
  std::vector<unsigned> input_sizes_;                 // total length of each input vector
  std::unordered_map<std::string, PreprocessParams> prep_info_map_;  // preprocessing info for each input group

  FloatArrays data_;

  bool debug_ = false;
  bool produceValueMap_;
  edm::Handle<edm::View<reco::Jet>> jets;
};

BoostedJetONNXJetTagsProducer::BoostedJetONNXJetTagsProducer(const edm::ParameterSet &iConfig, const ONNXRuntime *cache)
    : src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      debug_(iConfig.getUntrackedParameter<bool>("debugMode", false)),
      produceValueMap_(iConfig.getUntrackedParameter<bool>("produceValueMap", false)) {
  if (produceValueMap_) {
    jet_token_ = consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jets"));
  }

  ParticleNetConstructor(iConfig, true, input_names_, prep_info_map_, input_shapes_, input_sizes_, &data_);

  if (debug_) {
    LogDebug("BoostedJetONNXJetTagsProducer") << "<BoostedJetONNXJetTagsProducer::produce>:" << std::endl;
    for (unsigned i = 0; i < input_names_.size(); ++i) {
      const auto &group_name = input_names_.at(i);
      if (!input_shapes_.empty()) {
        LogDebug("BoostedJetONNXJetTagsProducer") << group_name << "\nshapes: ";
        for (const auto &x : input_shapes_.at(i)) {
          LogDebug("BoostedJetONNXJetTagsProducer") << x << ", ";
        }
      }
      LogDebug("BoostedJetONNXJetTagsProducer") << "\nvariables: ";
      for (const auto &x : prep_info_map_.at(group_name).var_names) {
        LogDebug("BoostedJetONNXJetTagsProducer") << x << ", ";
      }
      LogDebug("BoostedJetONNXJetTagsProducer") << "\n";
    }
    LogDebug("BoostedJetONNXJetTagsProducer") << "flav_names: ";
    for (const auto &flav_name : flav_names_) {
      LogDebug("BoostedJetONNXJetTagsProducer") << flav_name << ", ";
    }
    LogDebug("BoostedJetONNXJetTagsProducer") << "\n";
  }

  // get output names from flav_names
  for (const auto &flav_name : flav_names_) {
    if (!produceValueMap_) {
      produces<JetTagCollection>(flav_name);
    } else {
      produces<edm::ValueMap<float>>(flav_name);
    }
  }
}

BoostedJetONNXJetTagsProducer::~BoostedJetONNXJetTagsProducer() {}

void BoostedJetONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // pfDeepBoostedJetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfDeepBoostedJetTagInfos"));
  desc.add<std::string>("preprocess_json", "");
  // `preprocessParams` is deprecated -- use the preprocessing json file instead
  edm::ParameterSetDescription preprocessParams;
  preprocessParams.setAllowAnything();
  preprocessParams.setComment("`preprocessParams` is deprecated, please use `preprocess_json` instead.");
  desc.addOptional<edm::ParameterSetDescription>("preprocessParams", preprocessParams);
  desc.add<edm::FileInPath>("model_path",
                            edm::FileInPath("RecoBTag/Combined/data/DeepBoostedJet/V02/full/resnet.onnx"));
  desc.add<std::vector<std::string>>("flav_names",
                                     std::vector<std::string>{
                                         "probTbcq",
                                         "probTbqq",
                                         "probTbc",
                                         "probTbq",
                                         "probWcq",
                                         "probWqq",
                                         "probZbb",
                                         "probZcc",
                                         "probZqq",
                                         "probHbb",
                                         "probHcc",
                                         "probHqqqq",
                                         "probQCDbb",
                                         "probQCDcc",
                                         "probQCDb",
                                         "probQCDc",
                                         "probQCDothers",
                                     });
  desc.add<edm::InputTag>("jets", edm::InputTag(""));
  desc.addOptionalUntracked<bool>("produceValueMap", false);
  desc.addOptionalUntracked<bool>("debugMode", false);

  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<ONNXRuntime> BoostedJetONNXJetTagsProducer::initializeGlobalCache(const edm::ParameterSet &iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void BoostedJetONNXJetTagsProducer::globalEndJob(const ONNXRuntime *cache) {}

void BoostedJetONNXJetTagsProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(src_, tag_infos);
  if (produceValueMap_) {
    jets = iEvent.getHandle(jet_token_);
  }

  // initialize output collection
  std::vector<std::unique_ptr<JetTagCollection>> output_tags;
  std::vector<std::vector<float>> output_scores(flav_names_.size(), std::vector<float>(tag_infos->size(), -1.0));
  if (!tag_infos->empty()) {
    auto jet_ref = tag_infos->begin()->jet();
    auto ref2prod = edm::makeRefToBaseProdFrom(jet_ref, iEvent);
    for (std::size_t i = 0; i < flav_names_.size(); i++) {
      output_tags.emplace_back(std::make_unique<JetTagCollection>(ref2prod));
    }
  } else {
    for (std::size_t i = 0; i < flav_names_.size(); i++) {
      output_tags.emplace_back(std::make_unique<JetTagCollection>());
    }
  }

  for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
    const auto &taginfo = (*tag_infos)[jet_n];
    std::vector<float> outputs(flav_names_.size(), 0);  // init as all zeros

    if (!taginfo.features().empty()) {
      // convert inputs
      make_inputs(taginfo);
      // run prediction and get outputs
      outputs = globalCache()->run(input_names_, data_, input_shapes_)[0];
      assert(outputs.size() == flav_names_.size());
    }

    const auto &jet_ref = tag_infos->at(jet_n).jet();
    for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
      (*(output_tags[flav_n]))[jet_ref] = outputs[flav_n];
      output_scores[flav_n][jet_n] = outputs[flav_n];
    }
  }

  if (debug_) {
    LogDebug("produce") << "<BoostedJetONNXJetTagsProducer::produce>:" << std::endl
                        << "=== " << iEvent.id().run() << ":" << iEvent.id().luminosityBlock() << ":"
                        << iEvent.id().event() << " ===" << std::endl;
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      const auto &jet_ref = tag_infos->at(jet_n).jet();
      LogDebug("produce") << " - Jet #" << jet_n << ", pt=" << jet_ref->pt() << ", eta=" << jet_ref->eta()
                          << ", phi=" << jet_ref->phi() << std::endl;
      for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
        if (!produceValueMap_) {
          LogDebug("produce") << "    " << flav_names_.at(flav_n) << " = " << (*(output_tags.at(flav_n)))[jet_ref]
                              << std::endl;
        } else {
          LogDebug("produce") << "    " << flav_names_.at(flav_n) << " = " << output_scores[flav_n][jet_n] << std::endl;
        }
      }
    }
  }

  // put into the event
  if (!produceValueMap_) {
    for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
      iEvent.put(std::move(output_tags[flav_n]), flav_names_[flav_n]);
    }
  } else {
    for (size_t k = 0; k < output_scores.size(); k++) {
      std::unique_ptr<edm::ValueMap<float>> VM(new edm::ValueMap<float>());
      edm::ValueMap<float>::Filler filler(*VM);
      filler.insert(jets, output_scores.at(k).begin(), output_scores.at(k).end());
      filler.fill();
      iEvent.put(std::move(VM), flav_names_[k]);
    }
  }
}
void BoostedJetONNXJetTagsProducer::make_inputs(const reco::DeepBoostedJetTagInfo &taginfo) {
  for (unsigned igroup = 0; igroup < input_names_.size(); ++igroup) {
    const auto &group_name = input_names_[igroup];
    const auto &prep_params = prep_info_map_.at(group_name);
    auto &group_values = data_[igroup];
    group_values.resize(input_sizes_[igroup]);
    // first reset group_values to 0
    std::fill(group_values.begin(), group_values.end(), 0);
    unsigned curr_pos = 0;
    // transform/pad
    for (unsigned i = 0; i < prep_params.var_names.size(); ++i) {
      const auto &varname = prep_params.var_names[i];
      const auto &raw_value = taginfo.features().get(varname);
      const auto &info = prep_params.info(varname);
      int insize = center_norm_pad(raw_value,
                                   info.center,
                                   info.norm_factor,
                                   prep_params.min_length,
                                   prep_params.max_length,
                                   group_values,
                                   curr_pos,
                                   info.pad,
                                   info.replace_inf_value,
                                   info.lower_bound,
                                   info.upper_bound);
      curr_pos += insize;
      if (i == 0 && (!input_shapes_.empty())) {
        input_shapes_[igroup][2] = insize;
      }

      if (debug_) {
        LogDebug("make_inputs") << "<BoostedJetONNXJetTagsProducer::make_inputs>:" << std::endl
                                << " -- var=" << varname << ", center=" << info.center << ", scale=" << info.norm_factor
                                << ", replace=" << info.replace_inf_value << ", pad=" << info.pad << std::endl;
        for (unsigned i = curr_pos - insize; i < curr_pos; i++) {
          LogDebug("make_inputs") << group_values[i] << ",";
        }
        LogDebug("make_inputs") << std::endl;
      }
    }
    group_values.resize(curr_pos);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(BoostedJetONNXJetTagsProducer);
