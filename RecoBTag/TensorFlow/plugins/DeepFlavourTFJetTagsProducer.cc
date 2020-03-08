#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"
#include "RecoBTag/FeatureTools/interface/tensor_fillers.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class DeepFlavourTFJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<tensorflow::GraphDef>> {
public:
  explicit DeepFlavourTFJetTagsProducer(const edm::ParameterSet&, const tensorflow::GraphDef*);
  ~DeepFlavourTFJetTagsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<tensorflow::GraphDef> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const tensorflow::GraphDef*);

private:
  typedef std::vector<reco::DeepFlavourTagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  void produce(edm::Event&, const edm::EventSetup&) override;

  void make_inputs(unsigned i_jet, const reco::DeepFlavourTagInfo& taginfo);

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  enum InputIndexes { kGlobal = 0, kChargedCandidates = 1, kNeutralCandidates = 2, kVertices = 3, kJetPt = 4 };
  constexpr static unsigned n_features_global_ = 15;
  constexpr static unsigned n_cpf_ = 25;
  constexpr static unsigned n_features_cpf_ = 16;
  constexpr static unsigned n_npf_ = 25;
  constexpr static unsigned n_features_npf_ = 6;
  constexpr static unsigned n_sv_ = 4;
  constexpr static unsigned n_features_sv_ = 12;
  constexpr static unsigned n_features_jetpt_ = 1;
  const static std::vector<tensorflow::TensorShape> input_shapes_;

  // hold the input data
  tensorflow::NamedTensorList data_;
  // session for TF evaluation
  tensorflow::Session* session_ = nullptr;
};

const std::vector<tensorflow::TensorShape> DeepFlavourTFJetTagsProducer::input_shapes_{
    {1, n_features_global_},
    {1, n_cpf_, n_features_cpf_},
    {1, n_npf_, n_features_npf_},
    {1, n_sv_, n_features_sv_},
    {1, n_features_jetpt_},
};

DeepFlavourTFJetTagsProducer::DeepFlavourTFJetTagsProducer(const edm::ParameterSet& iConfig,
                                                           const tensorflow::GraphDef* cache)
    : src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
      output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")) {
  assert(input_names_.size() == input_shapes_.size());
  for (const auto& name : input_names_) {
    data_.emplace_back(name, tensorflow::Tensor());
  }

  for (const auto& name : iConfig.getParameter<std::vector<std::string>>("lp_names")) {
    data_.emplace_back(name, tensorflow::Tensor(tensorflow::DT_BOOL, {}));
    data_.back().second.scalar<bool>()() = false;
  }

  // get threading config and build session options
  size_t nThreads = iConfig.getParameter<unsigned int>("nThreads");
  std::string singleThreadPool = iConfig.getParameter<std::string>("singleThreadPool");
  tensorflow::SessionOptions sessionOptions;
  tensorflow::setThreading(sessionOptions, nThreads, singleThreadPool);

  // create the session using the meta graph from the cache
  session_ = tensorflow::createSession(cache, sessionOptions);

  // get output names from flav_names
  for (const auto& flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }
}

DeepFlavourTFJetTagsProducer::~DeepFlavourTFJetTagsProducer() {
  // close and delete the session
  if (session_ != nullptr) {
    tensorflow::closeSession(session_);
  }
}

void DeepFlavourTFJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfDeepFlavourJetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfDeepFlavourTagInfos"));
  desc.add<std::vector<std::string>>("input_names", {"input_1", "input_2", "input_3", "input_4", "input_5"});
  desc.add<std::vector<std::string>>("lp_names", {});
  desc.add<edm::FileInPath>("model_path",
                            edm::FileInPath("RecoBTag/Combined/data/DeepFlavourV03_10X_training/constant_graph.pb"));
  desc.add<std::vector<std::string>>("output_names", {"ID_pred/Softmax"});
  desc.add<std::vector<std::string>>(
      "flav_names", std::vector<std::string>{"probb", "probbb", "problepb", "probc", "probuds", "probg"});
  desc.add<unsigned int>("nThreads", 1);
  desc.add<std::string>("singleThreadPool", "no_threads");

  descriptions.add("pfDeepFlavourTFJetTags", desc);
}

std::unique_ptr<tensorflow::GraphDef> DeepFlavourTFJetTagsProducer::initializeGlobalCache(
    const edm::ParameterSet& iConfig) {
  // set the tensorflow log level to error
  tensorflow::setLogging("3");
  return std::unique_ptr<tensorflow::GraphDef>(
      tensorflow::loadGraphDef(iConfig.getParameter<edm::FileInPath>("model_path").fullPath()));
}

void DeepFlavourTFJetTagsProducer::globalEndJob(const tensorflow::GraphDef* cache) {}

void DeepFlavourTFJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(src_, tag_infos);

  std::vector<std::unique_ptr<JetTagCollection>> output_tags;
  if (!tag_infos->empty()) {
    // initialize output collection
    auto jet_ref = tag_infos->begin()->jet();
    auto ref2prod = edm::makeRefToBaseProdFrom(jet_ref, iEvent);
    for (std::size_t i = 0; i < flav_names_.size(); i++) {
      output_tags.emplace_back(std::make_unique<JetTagCollection>(ref2prod));
    }

    // init data storage
    for (unsigned i = 0; i < input_names_.size(); ++i) {
      auto shape = input_shapes_[i];
      shape.set_dim(0, tag_infos->size());
      data_[i].second = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
      data_[i].second.flat<float>().setZero();
    }

    // convert inputs
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      const auto& taginfo = (*tag_infos)[jet_n];
      make_inputs(jet_n, taginfo);
    }

    // run prediction
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::run(session_, data_, output_names_, &outputs);

    // get the outputs
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      const auto& jet_ref = tag_infos->at(jet_n).jet();
      for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
        (*(output_tags[flav_n]))[jet_ref] = outputs.at(0).matrix<float>()(jet_n, flav_n);
      }
    }
  } else {
    // create empty output collection
    for (std::size_t i = 0; i < flav_names_.size(); i++) {
      output_tags.emplace_back(std::make_unique<JetTagCollection>());
    }
  }

  // put into the event
  for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
    iEvent.put(std::move(output_tags[flav_n]), flav_names_[flav_n]);
  }
}

void DeepFlavourTFJetTagsProducer::make_inputs(unsigned i_jet, const reco::DeepFlavourTagInfo& taginfo) {
  const auto& features = taginfo.features();

  // jet and other global features
  btagbtvdeep::jet_tensor_filler(&data_[kGlobal].second.matrix<float>()(i_jet, 0), features, n_features_global_);

  // c_pf candidates
  auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)n_cpf_);
  btagbtvdeep::c_pf_tensor_filler(&data_[kChargedCandidates].second.tensor<float, 3>()(i_jet, 0, 0),
                                  max_c_pf_n,
                                  features.c_pf_features,
                                  n_features_cpf_);

  // n_pf candidates
  auto max_n_pf_n = std::min(features.n_pf_features.size(), (std::size_t)n_npf_);
  btagbtvdeep::n_pf_tensor_filler(&data_[kNeutralCandidates].second.tensor<float, 3>()(i_jet, 0, 0),
                                  max_n_pf_n,
                                  features.n_pf_features,
                                  n_features_npf_);

  // sv candidates
  auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)n_sv_);
  btagbtvdeep::sv_tensor_filler(
      &data_[kVertices].second.tensor<float, 3>()(i_jet, 0, 0), max_sv_n, features.sv_features, n_features_sv_);

  // last input: jet pt
  data_[kJetPt].second.matrix<float>()(i_jet, 0) = features.jet_features.pt;
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepFlavourTFJetTagsProducer);
