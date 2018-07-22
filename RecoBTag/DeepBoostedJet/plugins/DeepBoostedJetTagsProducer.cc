#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/DeepBoostedJetTagInfo.h"

#include "PhysicsTools/MXNet/interface/MXNetCppPredictor.h"

#include <iostream>
#include <fstream>

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
      center(median), scale(upper==median ? 1 : upper-median) {}
    float center = 0;
    float scale = 1;
  };

  unsigned var_length = 0;
  std::vector<std::string> var_names;
  std::unordered_map<std::string, VarInfo> var_info_map;

  VarInfo get_info(const std::string &name) const {
    try {
      return var_info_map.at(name);
    }catch (const std::out_of_range &e) {
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
    std::vector<std::pair<std::string,std::vector<unsigned int>>> flav_pairs_;
    std::vector<std::string> input_names_; // names of each input group - the ordering is important!
    std::vector<std::vector<unsigned int>> input_shapes_; // shapes of each input group
    std::unordered_map<std::string, PreprocessParams> prep_info_map_; // preprocessing info for each input group

    std::vector<std::vector<float>> data_;
    std::unique_ptr<mxnet::cpp::MXNetCppPredictor> predictor_;

    bool debug_ = false;
};

DeepBoostedJetTagsProducer::DeepBoostedJetTagsProducer(const edm::ParameterSet& iConfig, const MXBlockCache* cache)
: src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src")))
, debug_(iConfig.getUntrackedParameter<bool>("debugMode", false))
{

  // load preprocessing info
  input_shapes_.clear();
  prep_info_map_.clear();
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
  }

  if (debug_) {
    for (unsigned i=0; i<input_names_.size(); ++i){
      const auto &group_name = input_names_.at(i);
      std::cerr << group_name << "\nshapes: ";
      for (const auto &x : input_shapes_.at(i)){
        std::cerr << x << ", ";
      }
      std::cerr << "\nvariables: ";
      for (const auto &x : prep_info_map_.at(group_name).var_names){
        std::cerr << x << ", ";
      }
      std::cerr << "\n";
    }
  }

  // init MXNetPredictor
  predictor_.reset(new mxnet::cpp::MXNetCppPredictor(*cache->block));
  predictor_->set_input_shapes(input_names_, input_shapes_);

  // get output names from flav_table
  const auto & flav_pset = iConfig.getParameter<edm::ParameterSet>("flav_table");
  for (const auto flav_pair : flav_pset.tbl()) {
    const auto & flav_name = flav_pair.first;
    flav_pairs_.emplace_back(flav_name,
                             flav_pset.getParameter<std::vector<unsigned int>>(flav_name));
  }

  for (const auto & flav_pair : flav_pairs_) {
    produces<JetTagCollection>(flav_pair.first);
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
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::vector<unsigned int>>("probTbcq",      {0});
    psd0.add<std::vector<unsigned int>>("probTbqq",      {1});
    psd0.add<std::vector<unsigned int>>("probTbc",       {2});
    psd0.add<std::vector<unsigned int>>("probTbq",       {3});
    psd0.add<std::vector<unsigned int>>("probWcq",       {4});
    psd0.add<std::vector<unsigned int>>("probWqq",       {5});
    psd0.add<std::vector<unsigned int>>("probZbb",       {6});
    psd0.add<std::vector<unsigned int>>("probZcc",       {7});
    psd0.add<std::vector<unsigned int>>("probZqq",       {8});
    psd0.add<std::vector<unsigned int>>("probHbb",       {9});
    psd0.add<std::vector<unsigned int>>("probHcc",       {10});
    psd0.add<std::vector<unsigned int>>("probHqqqq",     {11});
    psd0.add<std::vector<unsigned int>>("probQCDbb",     {12});
    psd0.add<std::vector<unsigned int>>("probQCDcc",     {13});
    psd0.add<std::vector<unsigned int>>("probQCDb",      {14});
    psd0.add<std::vector<unsigned int>>("probQCDc",      {15});
    psd0.add<std::vector<unsigned int>>("probQCDothers", {16});
    desc.add<edm::ParameterSetDescription>("flav_table", psd0);
  }
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
  for (std::size_t i=0; i < flav_pairs_.size(); i++) {
    if (!tag_infos->empty()) {
      auto jet_ref = tag_infos->begin()->jet();
      output_tags.emplace_back(std::make_unique<JetTagCollection>(
            edm::makeRefToBaseProdFrom(jet_ref, iEvent)));
    } else {
      output_tags.emplace_back(std::make_unique<JetTagCollection>());
    }
  }

  for (unsigned jet_n=0; jet_n<tag_infos->size(); ++jet_n){

    const auto& taginfo = tag_infos->at(jet_n);
    std::vector<float> outputs(flav_pairs_.size(), 0); // init as all zeros

    if (!taginfo.features().empty()){
      // convert inputs
      make_inputs(taginfo);
      // run prediction and get outputs
      outputs = predictor_->predict(data_);
    }

    const auto & jet_ref = tag_infos->at(jet_n).jet();
    for (std::size_t flav_n=0; flav_n < flav_pairs_.size(); flav_n++) {
      const auto & flav_pair = flav_pairs_.at(flav_n);
      float o_sum = 0.;
      for (const unsigned int & ind : flav_pair.second) {
        o_sum += outputs.at(ind);
      }
      (*(output_tags.at(flav_n)))[jet_ref] = o_sum;
    }

  }

  if (debug_){
    std::cerr << "=== " <<  iEvent.id().run() << ":" << iEvent.id().luminosityBlock() << ":" << iEvent.id().event() << " ===" << std::endl;
    for (unsigned jet_n=0; jet_n<tag_infos->size(); ++jet_n){
      const auto & jet_ref = tag_infos->at(jet_n).jet();
      std::cerr << " - Jet #" << jet_n << ", pt=" << jet_ref->pt() << ", eta=" << jet_ref->eta() << ", phi=" << jet_ref->phi() << std::endl;
      for (std::size_t flav_n=0; flav_n < flav_pairs_.size(); ++flav_n) {
        std::cerr << "    " << flav_pairs_.at(flav_n).first << " = " << (*(output_tags.at(flav_n)))[jet_ref] << std::endl;
      }
    }
  }

  // put into the event
  for (std::size_t i=0; i < flav_pairs_.size(); i++) {
    iEvent.put(std::move(output_tags[i]), flav_pairs_.at(i).first);
  }

}

std::vector<float> DeepBoostedJetTagsProducer::center_norm_pad(
    const std::vector<float>& input, float center, float scale,
    unsigned target_length, float pad_value, float min, float max) {
  // do variable shifting/scaling/padding/clipping in one go

  auto clip = [](float value, float low, float high){
    if (low >= high) throw cms::Exception("InvalidArgument") << "Error in clip: low >= high!";
    if (value < low) return low;
    if (value > high) return high;
    return value;
  };

  pad_value = clip(pad_value, min, max);
  std::vector<float> out(target_length, pad_value);
  for (unsigned i=0; i<input.size() && i<target_length; ++i){
    out.at(i) = (input.at(i) - center) / scale;
    if (min < max) out.at(i) = clip(out.at(i), min, max);
  }
  return out;

}

void DeepBoostedJetTagsProducer::make_inputs(const reco::DeepBoostedJetTagInfo& taginfo) {
  data_.clear();
  for (const auto &group_name : input_names_) {
    // initiate with an empty vector
    data_.emplace_back();
    auto &group_values = data_.back();
    const auto& prep_params = prep_info_map_.at(group_name);
    // transform/pad
    int var_ref_len = -1;
    for (const auto &varname : prep_params.var_names){
      const auto &raw_value = taginfo.features().get(varname);
      // check consistency of the variable length
      if (var_ref_len == -1) {
        var_ref_len = raw_value.size();
      } else {
        if (static_cast<int>(raw_value.size()) != var_ref_len)
          throw cms::Exception("InvalidArgument") << "Inconsistent variable length " << raw_value.size() << " for " << varname << ", should be " << var_ref_len;
      }
      const auto &info = prep_params.get_info(varname);
      float pad = 0; // pad w/ zero
      auto val = center_norm_pad(raw_value, info.center, info.scale, prep_params.var_length, pad, -5, 5);
      group_values.insert(group_values.end(), val.begin(), val.end());

      if (debug_){
        std::cerr << " -- var=" << varname << ", center=" << info.center << ", scale=" << info.scale << ", pad=" << pad << std::endl;
        std::cerr << "values (first 7 and last 3): " << val.at(0) << ", " << val.at(1) << ", " << val.at(2) << ", " << val.at(3) << ", " << val.at(4) << ", " << val.at(5) << ", " << val.at(6) << " ... "
            << val.at(prep_params.var_length-3) << ", " << val.at(prep_params.var_length-2) << ", " << val.at(prep_params.var_length-1) << std::endl;
      }

    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepBoostedJetTagsProducer);
