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

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

using namespace cms::Ort;

// struct to hold preprocessing parameters
struct PreprocessParams {
  struct VarInfo {
    VarInfo() {}
    VarInfo(float median, float norm_factor) : center(median), norm_factor(norm_factor) {}
    float center = 0;
    float norm_factor = 1;
  };

  unsigned var_length = 0;
  std::vector<std::string> var_names;
  std::unordered_map<std::string, VarInfo> var_info_map;

  VarInfo get_info(const std::string &name) const {
    auto item = var_info_map.find(name);
    if (item != var_info_map.end()) {
      return item->second;
    } else {
      throw cms::Exception("InvalidArgument") << "Cannot find variable info for " << name;
    }
  }
};

class DeepBoostedJetONNXJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit DeepBoostedJetONNXJetTagsProducer(const edm::ParameterSet &, const ONNXRuntime *);
  ~DeepBoostedJetONNXJetTagsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &);
  static void globalEndJob(const ONNXRuntime *);

private:
  typedef std::vector<reco::DeepBoostedJetTagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  void produce(edm::Event &, const edm::EventSetup &) override;

  std::vector<float> center_norm_pad(const std::vector<float> &input,
                                     float center,
                                     float scale,
                                     unsigned target_length,
                                     float pad_value = 0,
                                     float min = 0,
                                     float max = -1);
  void make_inputs(const reco::DeepBoostedJetTagInfo &taginfo);

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;                  // names of the output scores
  std::vector<std::string> input_names_;                 // names of each input group - the ordering is important!
  std::vector<std::vector<unsigned int>> input_shapes_;  // shapes of each input group
  std::unordered_map<std::string, PreprocessParams> prep_info_map_;  // preprocessing info for each input group

  FloatArrays data_;

  bool debug_ = false;
};

DeepBoostedJetONNXJetTagsProducer::DeepBoostedJetONNXJetTagsProducer(const edm::ParameterSet &iConfig,
                                                                     const ONNXRuntime *cache)
    : src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      debug_(iConfig.getUntrackedParameter<bool>("debugMode", false)) {
  // load preprocessing info
  const auto &prep_pset = iConfig.getParameterSet("preprocessParams");
  input_names_ = prep_pset.getParameter<std::vector<std::string>>("input_names");
  for (const auto &group_name : input_names_) {
    const auto &group_pset = prep_pset.getParameterSet(group_name);
    input_shapes_.push_back(group_pset.getParameter<std::vector<unsigned>>("input_shape"));
    auto &prep_params = prep_info_map_[group_name];
    prep_params.var_length = group_pset.getParameter<unsigned>("var_length");
    prep_params.var_names = group_pset.getParameter<std::vector<std::string>>("var_names");
    const auto &var_info_pset = group_pset.getParameterSet("var_infos");
    for (const auto &var_name : prep_params.var_names) {
      const auto &var_pset = var_info_pset.getParameterSet(var_name);
      double median = var_pset.getParameter<double>("median");
      double norm_factor = var_pset.getParameter<double>("norm_factor");
      prep_params.var_info_map[var_name] = PreprocessParams::VarInfo(median, norm_factor);
    }

    // create data storage with a fixed size vector initilized w/ 0
    unsigned len = prep_params.var_length * prep_params.var_names.size();
    data_.emplace_back(len, 0);
  }

  if (debug_) {
    for (unsigned i = 0; i < input_names_.size(); ++i) {
      const auto &group_name = input_names_.at(i);
      std::cout << group_name << "\nshapes: ";
      for (const auto &x : input_shapes_.at(i)) {
        std::cout << x << ", ";
      }
      std::cout << "\nvariables: ";
      for (const auto &x : prep_info_map_.at(group_name).var_names) {
        std::cout << x << ", ";
      }
      std::cout << "\n";
    }
  }

  // get output names from flav_names
  for (const auto &flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }
}

DeepBoostedJetONNXJetTagsProducer::~DeepBoostedJetONNXJetTagsProducer() {}

void DeepBoostedJetONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // pfDeepBoostedJetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfDeepBoostedJetTagInfos"));
  edm::ParameterSetDescription preprocessParams;
  preprocessParams.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("preprocessParams", preprocessParams);
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
  desc.addOptionalUntracked<bool>("debugMode", false);

  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<ONNXRuntime> DeepBoostedJetONNXJetTagsProducer::initializeGlobalCache(const edm::ParameterSet &iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void DeepBoostedJetONNXJetTagsProducer::globalEndJob(const ONNXRuntime *cache) {}

void DeepBoostedJetONNXJetTagsProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(src_, tag_infos);

  // initialize output collection
  std::vector<std::unique_ptr<JetTagCollection>> output_tags;
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
      outputs = globalCache()->run(input_names_, data_)[0];
      assert(outputs.size() == flav_names_.size());
    }

    const auto &jet_ref = tag_infos->at(jet_n).jet();
    for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
      (*(output_tags[flav_n]))[jet_ref] = outputs[flav_n];
    }
  }

  if (debug_) {
    std::cout << "=== " << iEvent.id().run() << ":" << iEvent.id().luminosityBlock() << ":" << iEvent.id().event()
              << " ===" << std::endl;
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      const auto &jet_ref = tag_infos->at(jet_n).jet();
      std::cout << " - Jet #" << jet_n << ", pt=" << jet_ref->pt() << ", eta=" << jet_ref->eta()
                << ", phi=" << jet_ref->phi() << std::endl;
      for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
        std::cout << "    " << flav_names_.at(flav_n) << " = " << (*(output_tags.at(flav_n)))[jet_ref] << std::endl;
      }
    }
  }

  // put into the event
  for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
    iEvent.put(std::move(output_tags[flav_n]), flav_names_[flav_n]);
  }
}

std::vector<float> DeepBoostedJetONNXJetTagsProducer::center_norm_pad(const std::vector<float> &input,
                                                                      float center,
                                                                      float norm_factor,
                                                                      unsigned target_length,
                                                                      float pad_value,
                                                                      float min,
                                                                      float max) {
  // do variable shifting/scaling/padding/clipping in one go

  assert(min <= pad_value && pad_value <= max);

  std::vector<float> out(target_length, pad_value);
  for (unsigned i = 0; i < input.size() && i < target_length; ++i) {
    out[i] = std::clamp((input[i] - center) * norm_factor, min, max);
  }
  return out;
}

void DeepBoostedJetONNXJetTagsProducer::make_inputs(const reco::DeepBoostedJetTagInfo &taginfo) {
  for (unsigned igroup = 0; igroup < input_names_.size(); ++igroup) {
    const auto &group_name = input_names_[igroup];
    auto &group_values = data_[igroup];
    const auto &prep_params = prep_info_map_.at(group_name);
    // first reset group_values to 0
    std::fill(group_values.begin(), group_values.end(), 0);
    unsigned curr_pos = 0;
    // transform/pad
    for (const auto &varname : prep_params.var_names) {
      const auto &raw_value = taginfo.features().get(varname);
      const auto &info = prep_params.get_info(varname);
      const float pad = 0;  // pad w/ zero
      auto val = center_norm_pad(raw_value, info.center, info.norm_factor, prep_params.var_length, pad, -5, 5);
      std::copy(val.begin(), val.end(), group_values.begin() + curr_pos);
      curr_pos += prep_params.var_length;

      if (debug_) {
        std::cout << " -- var=" << varname << ", center=" << info.center << ", scale=" << info.norm_factor
                  << ", pad=" << pad << std::endl;
        std::cout << "values (first 7 and last 3): " << val.at(0) << ", " << val.at(1) << ", " << val.at(2) << ", "
                  << val.at(3) << ", " << val.at(4) << ", " << val.at(5) << ", " << val.at(6) << " ... "
                  << val.at(prep_params.var_length - 3) << ", " << val.at(prep_params.var_length - 2) << ", "
                  << val.at(prep_params.var_length - 1) << std::endl;
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepBoostedJetONNXJetTagsProducer);
