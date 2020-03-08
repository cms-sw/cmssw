#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/DeepDoubleXTagInfo.h"
#include "RecoBTag/FeatureTools/interface/tensor_fillers.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include <algorithm>

class DeepDoubleXTFJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<tensorflow::GraphDef>> {
public:
  explicit DeepDoubleXTFJetTagsProducer(const edm::ParameterSet&, const tensorflow::GraphDef*);
  ~DeepDoubleXTFJetTagsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<tensorflow::GraphDef> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const tensorflow::GraphDef*);

private:
  typedef std::vector<reco::DeepDoubleXTagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  void produce(edm::Event&, const edm::EventSetup&) override;

  void make_inputs(unsigned i_jet, const reco::DeepDoubleXTagInfo& taginfo);

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  enum InputIndexes { kGlobal = 0, kChargedCandidates = 1, kVertices = 2 };
  constexpr static unsigned n_features_global_ = 27;
  constexpr static unsigned n_cpf_ = 60;
  constexpr static unsigned n_features_cpf_ = 8;
  constexpr static unsigned n_sv_ = 5;
  constexpr static unsigned n_features_sv_ = 2;
  const static std::vector<tensorflow::TensorShape> input_shapes_;

  // hold the input data
  tensorflow::NamedTensorList data_;
  // session for TF evaluation
  tensorflow::Session* session_ = nullptr;
};

const std::vector<tensorflow::TensorShape> DeepDoubleXTFJetTagsProducer::input_shapes_{
    {1, n_features_global_},
    {1, n_cpf_, n_features_cpf_},
    {1, n_sv_, n_features_sv_},
};

DeepDoubleXTFJetTagsProducer::DeepDoubleXTFJetTagsProducer(const edm::ParameterSet& iConfig,
                                                           const tensorflow::GraphDef* cache)
    : src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
      output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")) {
  assert(input_names_.size() == input_shapes_.size());
  for (const auto& name : input_names_) {
    data_.emplace_back(name, tensorflow::Tensor());
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

DeepDoubleXTFJetTagsProducer::~DeepDoubleXTFJetTagsProducer() {
  // close and delete the session
  if (session_ != nullptr) {
    tensorflow::closeSession(session_);
  }
}

void DeepDoubleXTFJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfDeepDoubleBvLJetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfDeepDoubleXTagInfos"));
  desc.add<std::vector<std::string>>("input_names", {"input_1", "input_2", "input_3"});
  desc.add<std::vector<std::string>>("output_names", {"ID_pred/Softmax"});
  desc.add<unsigned int>("nThreads", 1);
  desc.add<std::string>("singleThreadPool", "no_threads");

  using FIP = edm::FileInPath;
  using PDFIP = edm::ParameterDescription<edm::FileInPath>;
  using PDPSD = edm::ParameterDescription<std::vector<std::string>>;
  using PDCases = edm::ParameterDescriptionCases<std::string>;
  auto flavorCases = [&]() {
    return "BvL" >> (PDPSD("flav_names", std::vector<std::string>{"probQCD", "probHbb"}, true) and
                     PDFIP("model_path", FIP("RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDB.pb"), true)) or
           "CvL" >> (PDPSD("flav_names", std::vector<std::string>{"probQCD", "probHcc"}, true) and
                     PDFIP("model_path", FIP("RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDC.pb"), true)) or
           "CvB" >> (PDPSD("flav_names", std::vector<std::string>{"probHbb", "probHcc"}, true) and
                     PDFIP("model_path", FIP("RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDCvB.pb"), true));
  };
  auto descBvL(desc);
  descBvL.ifValue(edm::ParameterDescription<std::string>("flavor", "BvL", true), flavorCases());
  descriptions.add("pfDeepDoubleBvLTFJetTags", descBvL);

  auto descCvL(desc);
  descCvL.ifValue(edm::ParameterDescription<std::string>("flavor", "CvL", true), flavorCases());
  descriptions.add("pfDeepDoubleCvLTFJetTags", descCvL);

  auto descCvB(desc);
  descCvB.ifValue(edm::ParameterDescription<std::string>("flavor", "CvB", true), flavorCases());
  descriptions.add("pfDeepDoubleCvBTFJetTags", descCvB);
}

std::unique_ptr<tensorflow::GraphDef> DeepDoubleXTFJetTagsProducer::initializeGlobalCache(
    const edm::ParameterSet& iConfig) {
  // set the tensorflow log level to error
  tensorflow::setLogging("3");
  return std::unique_ptr<tensorflow::GraphDef>(
      tensorflow::loadGraphDef(iConfig.getParameter<edm::FileInPath>("model_path").fullPath()));
}

void DeepDoubleXTFJetTagsProducer::globalEndJob(const tensorflow::GraphDef* cache) {}

void DeepDoubleXTFJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

    // only need to run on jets with non-empty features
    auto batch_size = std::count_if(
        tag_infos->begin(), tag_infos->end(), [](const auto& taginfo) { return !taginfo.features().empty(); });

    std::vector<tensorflow::Tensor> outputs;
    if (batch_size > 0) {
      // init data storage
      for (unsigned i = 0; i < input_names_.size(); ++i) {
        auto shape = input_shapes_[i];
        shape.set_dim(0, batch_size);
        data_[i].second = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
        data_[i].second.flat<float>().setZero();
      }

      // convert inputs
      unsigned idx = 0;
      for (const auto& taginfo : *tag_infos) {
        if (!taginfo.features().empty()) {
          make_inputs(idx, taginfo);
          ++idx;
        }
      }

      // run prediction
      tensorflow::run(session_, data_, output_names_, &outputs);
    }

    // get the outputs
    unsigned idx = 0;
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      const auto& taginfo = tag_infos->at(jet_n);
      const auto& jet_ref = taginfo.jet();
      if (!taginfo.features().empty()) {
        for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
          (*(output_tags[flav_n]))[jet_ref] = outputs.at(0).matrix<float>()(idx, flav_n);
        }
        ++idx;
      } else {
        for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
          (*(output_tags[flav_n]))[jet_ref] = -1.;
        }
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

void DeepDoubleXTFJetTagsProducer::make_inputs(unsigned i_jet, const reco::DeepDoubleXTagInfo& taginfo) {
  const auto& features = taginfo.features();

  // DoubleB features
  btagbtvdeep::db_tensor_filler(&data_[kGlobal].second.matrix<float>()(i_jet, 0), features, n_features_global_);

  // c_pf candidates
  auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)n_cpf_);
  btagbtvdeep::c_pf_reduced_tensor_filler(&data_[kChargedCandidates].second.tensor<float, 3>()(i_jet, 0, 0),
                                          max_c_pf_n,
                                          features.c_pf_features,
                                          n_features_cpf_);

  // sv candidates
  auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)n_sv_);
  btagbtvdeep::sv_reduced_tensor_filler(
      &data_[kVertices].second.tensor<float, 3>()(i_jet, 0, 0), max_sv_n, features.sv_features, n_features_sv_);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepDoubleXTFJetTagsProducer);
