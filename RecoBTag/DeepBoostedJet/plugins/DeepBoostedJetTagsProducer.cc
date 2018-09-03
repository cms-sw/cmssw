#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/DeepBoostedJetTagInfo.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include "PhysicsTools/MXNet/interface/Predictor.h"

// Hold the mxnet model block (symbol + params) in the edm::GlobalCache.
struct MXBlockCache {
  MXBlockCache() : block(nullptr) {
  }

  std::atomic<mxnet::cpp::Block*> block;
};

// struct to hold preprocessing parameters
struct PreprocessParams {
  struct VarInfo {
    VarInfo() {}
    VarInfo(float median, float upper) :
      center(median), norm_factor(upper==median ? 1 : 1./(upper-median)) {}
    float center = 0;
    float norm_factor = 1;
  };

  unsigned var_length = 0;
  std::vector<std::string> var_names;
  std::unordered_map<std::string, VarInfo> var_info_map;

  VarInfo get_info(const std::string &name) const {
    auto item = var_info_map.find(name);
    if (item != var_info_map.end()){
      return item->second;
    } else {
      throw cms::Exception("InvalidArgument") << "Cannot find variable info for " << name;
    }
  }
};

class DeepBoostedJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<MXBlockCache>> {

  public:
    explicit DeepBoostedJetTagsProducer(const edm::ParameterSet&, const MXBlockCache*);
    ~DeepBoostedJetTagsProducer() override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

    static std::unique_ptr<MXBlockCache> initializeGlobalCache(const edm::ParameterSet&);
    static void globalEndJob(const MXBlockCache*);

  private:
    typedef std::vector<reco::DeepBoostedJetTagInfo> TagInfoCollection;
    typedef reco::JetTagCollection JetTagCollection;

    void beginStream(edm::StreamID) override {}
    void produce(edm::Event&, const edm::EventSetup&) override;
    void endStream() override {}

    std::vector<float> center_norm_pad(const std::vector<float>& input,
        float center, float scale,
        unsigned target_length, float pad_value=0,
        float min=0, float max=-1);
    void make_inputs(const reco::DeepBoostedJetTagInfo &taginfo);

    const edm::EDGetTokenT< TagInfoCollection > src_;
    std::vector<std::string> flav_names_;  // names of the output scores
    std::vector<std::string> input_names_; // names of each input group - the ordering is important!
    std::vector<std::vector<unsigned int>> input_shapes_; // shapes of each input group
    std::unordered_map<std::string, PreprocessParams> prep_info_map_; // preprocessing info for each input group

    std::vector<std::vector<float>> data_;
    std::unique_ptr<mxnet::cpp::Predictor> predictor_;

    bool debug_ = false;
};

DeepBoostedJetTagsProducer::DeepBoostedJetTagsProducer(const edm::ParameterSet& iConfig, const MXBlockCache* cache)
: src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src")))
, flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names"))
, debug_(iConfig.getUntrackedParameter<bool>("debugMode", false))
{

  // load preprocessing info
  const auto &prep_pset = iConfig.getParameterSet("preprocessParams");
  input_names_ = prep_pset.getParameter<std::vector<std::string>>("input_names");
  for (const auto &group_name : input_names_){
    const auto &group_pset = prep_pset.getParameterSet(group_name);
    input_shapes_.push_back(group_pset.getParameter<std::vector<unsigned>>("input_shape"));
    auto& prep_params = prep_info_map_[group_name];
    prep_params.var_length = group_pset.getParameter<unsigned>("var_length");
    prep_params.var_names = group_pset.getParameter<std::vector<std::string>>("var_names");
    const auto &var_info_pset = group_pset.getParameterSet("var_infos");
    for (const auto &var_name : prep_params.var_names){
      const auto &var_pset = var_info_pset.getParameterSet(var_name);
      double median = var_pset.getParameter<double>("median");
      double upper = var_pset.getParameter<double>("upper");
      prep_params.var_info_map[var_name] = PreprocessParams::VarInfo(median, upper);
    }

    // create data storage with a fixed size vector initilized w/ 0
    unsigned len = prep_params.var_length * prep_params.var_names.size();
    data_.emplace_back(len, 0);
  }

  if (debug_) {
    for (unsigned i=0; i<input_names_.size(); ++i){
      const auto &group_name = input_names_.at(i);
      std::cout << group_name << "\nshapes: ";
      for (const auto &x : input_shapes_.at(i)){
        std::cout << x << ", ";
      }
      std::cout << "\nvariables: ";
      for (const auto &x : prep_info_map_.at(group_name).var_names){
        std::cout << x << ", ";
      }
      std::cout << "\n";
    }
  }

  // init MXNetPredictor
  predictor_.reset(new mxnet::cpp::Predictor(*cache->block));
  predictor_->set_input_shapes(input_names_, input_shapes_);

  // get output names from flav_names
  for (const auto &flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }

}

DeepBoostedJetTagsProducer::~DeepBoostedJetTagsProducer(){
}

void DeepBoostedJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  // pfDeepBoostedJetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfDeepBoostedJetTagInfos"));
  edm::ParameterSetDescription preprocessParams;
  preprocessParams.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("preprocessParams", preprocessParams);
  desc.add<edm::FileInPath>("model_path",
    edm::FileInPath("RecoBTag/Combined/data/DeepBoostedJet/V01/full/resnet-symbol.json"));
  desc.add<edm::FileInPath>("param_path",
    edm::FileInPath("RecoBTag/Combined/data/DeepBoostedJet/V01/full/resnet-0000.params"));
  desc.add<std::vector<std::string>>("flav_names", std::vector<std::string>{
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

  descriptions.add("pfDeepBoostedJetTags", desc);
}

std::unique_ptr<MXBlockCache> DeepBoostedJetTagsProducer::initializeGlobalCache(
  const edm::ParameterSet& iConfig)
{
  // get the model files
  std::string model_file = iConfig.getParameter<edm::FileInPath>("model_path").fullPath();
  std::string param_file = iConfig.getParameter<edm::FileInPath>("param_path").fullPath();

  // load the model and params and save it in the cache
  MXBlockCache* cache = new MXBlockCache();
  cache->block = new mxnet::cpp::Block(model_file, param_file);
  return std::unique_ptr<MXBlockCache>(cache);
}

void DeepBoostedJetTagsProducer::globalEndJob(const MXBlockCache* cache)
{
  if (cache->block != nullptr) {
    delete cache->block;
  }
}

void DeepBoostedJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(src_, tag_infos);

  // initialize output collection
  std::vector<std::unique_ptr<JetTagCollection>> output_tags;
  if (!tag_infos->empty()) {
    auto jet_ref = tag_infos->begin()->jet();
    auto ref2prod = edm::makeRefToBaseProdFrom(jet_ref, iEvent);
    for (std::size_t i=0; i < flav_names_.size(); i++) {
      output_tags.emplace_back(std::make_unique<JetTagCollection>(ref2prod));
    }
  } else {
    for (std::size_t i=0; i < flav_names_.size(); i++) {
      output_tags.emplace_back(std::make_unique<JetTagCollection>());
    }
  }

  for (unsigned jet_n=0; jet_n<tag_infos->size(); ++jet_n){

    const auto& taginfo = (*tag_infos)[jet_n];
    std::vector<float> outputs(flav_names_.size(), 0); // init as all zeros

    if (!taginfo.features().empty()){
      // convert inputs
      make_inputs(taginfo);
      // run prediction and get outputs
      outputs = predictor_->predict(data_);
      assert(outputs.size() == flav_names_.size());
    }

    const auto & jet_ref = tag_infos->at(jet_n).jet();
    for (std::size_t flav_n=0; flav_n < flav_names_.size(); flav_n++) {
      (*(output_tags[flav_n]))[jet_ref] = outputs[flav_n];
    }

  }

  if (debug_){
    std::cout << "=== " <<  iEvent.id().run() << ":" << iEvent.id().luminosityBlock() << ":" << iEvent.id().event() << " ===" << std::endl;
    for (unsigned jet_n=0; jet_n<tag_infos->size(); ++jet_n){
      const auto & jet_ref = tag_infos->at(jet_n).jet();
      std::cout << " - Jet #" << jet_n << ", pt=" << jet_ref->pt() << ", eta=" << jet_ref->eta() << ", phi=" << jet_ref->phi() << std::endl;
      for (std::size_t flav_n=0; flav_n < flav_names_.size(); ++flav_n) {
        std::cout << "    " << flav_names_.at(flav_n) << " = " << (*(output_tags.at(flav_n)))[jet_ref] << std::endl;
      }
    }
  }

  // put into the event
  for (std::size_t flav_n=0; flav_n < flav_names_.size(); ++flav_n) {
    iEvent.put(std::move(output_tags[flav_n]), flav_names_[flav_n]);
  }

}

std::vector<float> DeepBoostedJetTagsProducer::center_norm_pad(
    const std::vector<float>& input, float center, float norm_factor,
    unsigned target_length, float pad_value, float min, float max) {
  // do variable shifting/scaling/padding/clipping in one go

  assert(min<=pad_value && pad_value<=max);

  std::vector<float> out(target_length, pad_value);
  for (unsigned i=0; i<input.size() && i<target_length; ++i){
    out[i] = std::clamp((input[i] - center) * norm_factor, min, max);
  }
  return out;

}

void DeepBoostedJetTagsProducer::make_inputs(const reco::DeepBoostedJetTagInfo& taginfo) {
  for (unsigned igroup = 0; igroup<input_names_.size(); ++igroup) {
    const auto &group_name = input_names_[igroup];
    auto &group_values = data_[igroup];
    const auto& prep_params = prep_info_map_.at(group_name);
    // first reset group_values to 0
    std::fill(group_values.begin(), group_values.end(), 0);
    unsigned curr_pos = 0;
    // transform/pad
    for (const auto &varname : prep_params.var_names){
      const auto &raw_value = taginfo.features().get(varname);
      const auto &info = prep_params.get_info(varname);
      const float pad = 0; // pad w/ zero
      auto val = center_norm_pad(raw_value, info.center, info.norm_factor, prep_params.var_length, pad, -5, 5);
      std::copy(val.begin(), val.end(), group_values.begin()+curr_pos);
      curr_pos += prep_params.var_length;

      if (debug_){
        std::cout << " -- var=" << varname << ", center=" << info.center << ", scale=" << info.norm_factor << ", pad=" << pad << std::endl;
        std::cout << "values (first 7 and last 3): " << val.at(0) << ", " << val.at(1) << ", " << val.at(2) << ", " << val.at(3) << ", " << val.at(4) << ", " << val.at(5) << ", " << val.at(6) << " ... "
            << val.at(prep_params.var_length-3) << ", " << val.at(prep_params.var_length-2) << ", " << val.at(prep_params.var_length-1) << std::endl;
      }

    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepBoostedJetTagsProducer);
