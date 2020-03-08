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

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

using namespace cms::Ort;

class DeepFlavourONNXJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit DeepFlavourONNXJetTagsProducer(const edm::ParameterSet&, const ONNXRuntime*);
  ~DeepFlavourONNXJetTagsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

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
  const static std::vector<unsigned> input_sizes_;

  // hold the input data
  FloatArrays data_;
};

const std::vector<unsigned> DeepFlavourONNXJetTagsProducer::input_sizes_{
    n_features_global_, n_cpf_* n_features_cpf_, n_npf_* n_features_npf_, n_sv_* n_features_sv_, n_features_jetpt_};

DeepFlavourONNXJetTagsProducer::DeepFlavourONNXJetTagsProducer(const edm::ParameterSet& iConfig,
                                                               const ONNXRuntime* cache)
    : src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
      output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")) {
  assert(input_names_.size() == input_sizes_.size());
  // get output names from flav_names
  for (const auto& flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }
}

DeepFlavourONNXJetTagsProducer::~DeepFlavourONNXJetTagsProducer() {}

void DeepFlavourONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfDeepFlavourJetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfDeepFlavourTagInfos"));
  desc.add<std::vector<std::string>>("input_names", {"input_1", "input_2", "input_3", "input_4", "input_5"});
  desc.add<edm::FileInPath>("model_path",
                            edm::FileInPath("RecoBTag/Combined/data/DeepFlavourV03_10X_training/model.onnx"));
  desc.add<std::vector<std::string>>("output_names", {"ID_pred/Softmax:0"});
  desc.add<std::vector<std::string>>(
      "flav_names", std::vector<std::string>{"probb", "probbb", "problepb", "probc", "probuds", "probg"});

  descriptions.add("pfDeepFlavourONNXJetTags", desc);
}

std::unique_ptr<ONNXRuntime> DeepFlavourONNXJetTagsProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void DeepFlavourONNXJetTagsProducer::globalEndJob(const ONNXRuntime* cache) {}

void DeepFlavourONNXJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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
    data_.clear();
    for (const auto& len : input_sizes_) {
      data_.emplace_back(tag_infos->size() * len, 0);
    }

    // convert inputs
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      const auto& taginfo = (*tag_infos)[jet_n];
      make_inputs(jet_n, taginfo);
    }

    // run prediction
    auto outputs = globalCache()->run(input_names_, data_, output_names_, tag_infos->size())[0];
    assert(outputs.size() == flav_names_.size() * tag_infos->size());

    // get the outputs
    unsigned i_output = 0;
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      const auto& jet_ref = tag_infos->at(jet_n).jet();
      for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
        (*(output_tags[flav_n]))[jet_ref] = outputs[i_output];
        ++i_output;
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

void DeepFlavourONNXJetTagsProducer::make_inputs(unsigned i_jet, const reco::DeepFlavourTagInfo& taginfo) {
  const auto& features = taginfo.features();
  unsigned offset = 0;

  // jet and other global features
  offset = i_jet * input_sizes_[kGlobal];
  btagbtvdeep::jet_tensor_filler(&data_[kGlobal][offset], features, n_features_global_);

  // c_pf candidates
  offset = i_jet * input_sizes_[kChargedCandidates];
  auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)n_cpf_);
  btagbtvdeep::c_pf_tensor_filler(
      &data_[kChargedCandidates][offset], max_c_pf_n, features.c_pf_features, n_features_cpf_);

  // n_pf candidates
  offset = i_jet * input_sizes_[kNeutralCandidates];
  auto max_n_pf_n = std::min(features.n_pf_features.size(), (std::size_t)n_npf_);
  btagbtvdeep::n_pf_tensor_filler(
      &data_[kNeutralCandidates][offset], max_n_pf_n, features.n_pf_features, n_features_npf_);

  // sv candidates
  offset = i_jet * input_sizes_[kVertices];
  auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)n_sv_);
  btagbtvdeep::sv_tensor_filler(&data_[kVertices][offset], max_sv_n, features.sv_features, n_features_sv_);

  // last input: jet pt
  offset = i_jet * input_sizes_[kJetPt];
  data_[kJetPt][offset] = features.jet_features.pt;
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepFlavourONNXJetTagsProducer);
