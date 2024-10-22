#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/ParticleTransformerAK4TagInfo.h"
#include "DataFormats/BTauReco/interface/ParticleTransformerAK4Features.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include "RecoBTag/ONNXRuntime/interface/tensor_fillers.h"
#include "RecoBTag/ONNXRuntime/interface/tensor_configs.h"

using namespace cms::Ort;

class ParticleTransformerAK4ONNXJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit ParticleTransformerAK4ONNXJetTagsProducer(const edm::ParameterSet&, const ONNXRuntime*);
  ~ParticleTransformerAK4ONNXJetTagsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

private:
  typedef std::vector<reco::ParticleTransformerAK4TagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  void produce(edm::Event&, const edm::EventSetup&) override;

  void make_inputs(btagbtvdeep::ParticleTransformerAK4Features features);
  void get_input_sizes(const reco::FeaturesTagInfo<btagbtvdeep::ParticleTransformerAK4Features> taginfo);

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  unsigned int n_cpf_;
  unsigned int n_npf_;
  unsigned int n_sv_;
  std::vector<unsigned> input_sizes_;
  std::vector<std::vector<int64_t>> input_shapes_;  // shapes of each input group (-1 for dynamic axis)

  // hold the input data
  FloatArrays data_;
};

ParticleTransformerAK4ONNXJetTagsProducer::ParticleTransformerAK4ONNXJetTagsProducer(const edm::ParameterSet& iConfig,
                                                                                     const ONNXRuntime* cache)
    : src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
      output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")) {
  // get output names from flav_names
  for (const auto& flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }
}

void ParticleTransformerAK4ONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfParticleTransformerAK4JetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfParticleTransformerAK4TagInfos"));
  desc.add<std::vector<std::string>>("input_names", {"input_1", "input_2", "input_3", "input_4", "input_5", "input_6"});
  desc.add<edm::FileInPath>("model_path",
                            edm::FileInPath("RecoBTag/Combined/data/RobustParTAK4/PUPPI/V00/modelfile/model.onnx"));
  desc.add<std::vector<std::string>>("output_names", {"softmax"});
  desc.add<std::vector<std::string>>(
      "flav_names", std::vector<std::string>{"probb", "probbb", "problepb", "probc", "probuds", "probg"});

  descriptions.add("pfParticleTransformerAK4JetTags", desc);
}

std::unique_ptr<ONNXRuntime> ParticleTransformerAK4ONNXJetTagsProducer::initializeGlobalCache(
    const edm::ParameterSet& iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void ParticleTransformerAK4ONNXJetTagsProducer::globalEndJob(const ONNXRuntime* cache) {}

void ParticleTransformerAK4ONNXJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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
    const auto& taginfo = (*tag_infos)[jet_n];
    std::vector<float> outputs(flav_names_.size(), -1.0);
    if (taginfo.features().is_filled) {
      get_input_sizes(taginfo);

      // run prediction with dynamic batch size per event
      input_shapes_ = {{(int64_t)1, (int64_t)n_cpf_, (int64_t)parT::N_InputFeatures.at(parT::kChargedCandidates)},
                       {(int64_t)1, (int64_t)n_npf_, (int64_t)parT::N_InputFeatures.at(parT::kNeutralCandidates)},
                       {(int64_t)1, (int64_t)n_sv_, (int64_t)parT::N_InputFeatures.at(parT::kVertices)},
                       {(int64_t)1, (int64_t)n_cpf_, (int64_t)parT::N_InputFeatures.at(parT::kChargedCandidates4Vec)},
                       {(int64_t)1, (int64_t)n_npf_, (int64_t)parT::N_InputFeatures.at(parT::kNeutralCandidates4Vec)},
                       {(int64_t)1, (int64_t)n_sv_, (int64_t)parT::N_InputFeatures.at(parT::kVertices4Vec)}};

      outputs = globalCache()->run(input_names_, data_, input_shapes_, output_names_, 1)[0];
      assert(outputs.size() == flav_names_.size());
    }

    const auto& jet_ref = tag_infos->at(jet_n).jet();
    for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
      (*(output_tags[flav_n]))[jet_ref] = outputs[flav_n];
    }
  }

  // put into the event
  for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
    iEvent.put(std::move(output_tags[flav_n]), flav_names_[flav_n]);
  }
}

void ParticleTransformerAK4ONNXJetTagsProducer::get_input_sizes(
    const reco::FeaturesTagInfo<btagbtvdeep::ParticleTransformerAK4Features> taginfo) {
  const auto& features = taginfo.features();

  n_cpf_ = std::clamp((unsigned int)features.c_pf_features.size(), (unsigned int)1, (unsigned int)parT::n_cpf_accept);
  n_npf_ = std::clamp((unsigned int)features.n_pf_features.size(), (unsigned int)1, (unsigned int)parT::n_npf_accept);
  n_sv_ = std::clamp((unsigned int)features.sv_features.size(), (unsigned int)1, (unsigned int)parT::n_sv_accept);

  input_sizes_ = {
      n_cpf_ * parT::N_InputFeatures.at(parT::kChargedCandidates),
      n_npf_ * parT::N_InputFeatures.at(parT::kNeutralCandidates),
      n_sv_ * parT::N_InputFeatures.at(parT::kVertices),
      n_cpf_ * parT::N_InputFeatures.at(parT::kChargedCandidates4Vec),
      n_npf_ * parT::N_InputFeatures.at(parT::kNeutralCandidates4Vec),
      n_sv_ * parT::N_InputFeatures.at(parT::kVertices4Vec),
  };
  // init data storage
  data_.clear();
  for (const auto& len : input_sizes_) {
    data_.emplace_back(1 * len, 0);
  }

  make_inputs(features);
}

void ParticleTransformerAK4ONNXJetTagsProducer::make_inputs(btagbtvdeep::ParticleTransformerAK4Features features) {
  //float* ptr = nullptr;
  const float* start = nullptr;
  unsigned offset = 0;

  auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)n_cpf_);
  auto max_n_pf_n = std::min(features.n_pf_features.size(), (std::size_t)n_npf_);
  auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)n_sv_);

  // c_pf candidates
  parT_tensor_filler(data_, parT::kChargedCandidates, features.c_pf_features, max_c_pf_n, start, offset);
  // n_pf candidates
  parT_tensor_filler(data_, parT::kNeutralCandidates, features.n_pf_features, max_n_pf_n, start, offset);
  // sv candidates
  parT_tensor_filler(data_, parT::kVertices, features.sv_features, max_sv_n, start, offset);
  // cpf pairwise features (4-vectors)
  parT_tensor_filler(data_, parT::kChargedCandidates4Vec, features.c_pf_features, max_c_pf_n, start, offset);
  // npf pairwise features (4-vectors)
  parT_tensor_filler(data_, parT::kNeutralCandidates4Vec, features.n_pf_features, max_n_pf_n, start, offset);
  // sv pairwise features (4-vectors)
  parT_tensor_filler(data_, parT::kVertices4Vec, features.sv_features, max_sv_n, start, offset);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ParticleTransformerAK4ONNXJetTagsProducer);
