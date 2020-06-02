#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/HiggsInteractionNetTagInfo.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

using namespace cms::Ort;

struct PreprocessParamsSimple {
  unsigned var_length = 0;
  float pad = 0;
  std::vector<std::string> var_names;
};

class HiggsInteractionNetONNXJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit HiggsInteractionNetONNXJetTagsProducer(const edm::ParameterSet &, const ONNXRuntime *);
  ~HiggsInteractionNetONNXJetTagsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &);
  static void globalEndJob(const ONNXRuntime *);

private:
  typedef std::vector<reco::HiggsInteractionNetTagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  void produce(edm::Event &, const edm::EventSetup &) override;

  void make_inputs(const reco::HiggsInteractionNetTagInfo &taginfo);

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;                  // names of the output scores
  std::vector<std::string> input_names_;                 // names of each input group - the ordering is important!
  std::vector<std::vector<unsigned int>> input_shapes_;  // shapes of each input group
  std::unordered_map<std::string, PreprocessParamsSimple> prep_info_map_;  // preprocessing info for each input group

  FloatArrays data_;

  bool debug_ = false;
};

HiggsInteractionNetONNXJetTagsProducer::HiggsInteractionNetONNXJetTagsProducer(const edm::ParameterSet &iConfig,
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
    prep_params.pad = group_pset.getParameter<double>("pad");

    // create data storage with a fixed size vector initilized w/ pad
    unsigned len = prep_params.var_length * prep_params.var_names.size();
    data_.emplace_back(len, prep_params.pad);
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

HiggsInteractionNetONNXJetTagsProducer::~HiggsInteractionNetONNXJetTagsProducer() {}

void HiggsInteractionNetONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // pfHiggsInteractionNetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfHiggsInteractionNetTagInfos"));
  edm::ParameterSetDescription preprocessParams;
  preprocessParams.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("preprocessParams", preprocessParams);
  desc.add<edm::FileInPath>("model_path", edm::FileInPath("RecoBTag/Combined/data/HiggsInteractionNet/V00/IN.onnx"));
  desc.add<std::vector<std::string>>("flav_names",
                                     std::vector<std::string>{
                                         "probQCD",
                                         "probHbb",
                                     });
  desc.addOptionalUntracked<bool>("debugMode", false);

  descriptions.add("pfHiggsInteractionNetTags", desc);
}

std::unique_ptr<ONNXRuntime> HiggsInteractionNetONNXJetTagsProducer::initializeGlobalCache(
    const edm::ParameterSet &iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void HiggsInteractionNetONNXJetTagsProducer::globalEndJob(const ONNXRuntime *cache) {}

void HiggsInteractionNetONNXJetTagsProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
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

void HiggsInteractionNetONNXJetTagsProducer::make_inputs(const reco::HiggsInteractionNetTagInfo &taginfo) {
  for (unsigned igroup = 0; igroup < input_names_.size(); ++igroup) {
    const auto &group_name = input_names_.at(igroup);
    auto &group_values = data_.at(igroup);
    const auto &prep_params = prep_info_map_.at(group_name);
    // first reset group_values to pad
    std::fill(group_values.begin(), group_values.end(), prep_params.pad);
    unsigned curr_pos = 0;
    // transform/pad
    for (const auto &varname : prep_params.var_names) {
      const auto &raw_value = taginfo.features().get(varname);
      for (unsigned i = 0; i < raw_value.size() && i < prep_params.var_length; ++i) {
        group_values[curr_pos + i] = raw_value[i];
      }
      if (debug_) {
        std::cout << "group_name=" << group_name << ", varname=" << varname << ", pad=" << prep_params.pad
                  << ", var_length=" << prep_params.var_length << std::endl;
        std::cout << "values (first 5 and last 3): " << group_values.at(curr_pos) << ", "
                  << group_values.at(curr_pos + 1) << ", " << group_values.at(curr_pos + 2) << ", "
                  << group_values.at(curr_pos + 3) << ", " << group_values.at(curr_pos + 4) << " ... "
                  << group_values.at(curr_pos + prep_params.var_length - 3) << ", "
                  << group_values.at(curr_pos + prep_params.var_length - 2) << ", "
                  << group_values.at(curr_pos + prep_params.var_length - 1) << std::endl;
      }
      curr_pos += prep_params.var_length;
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiggsInteractionNetONNXJetTagsProducer);
