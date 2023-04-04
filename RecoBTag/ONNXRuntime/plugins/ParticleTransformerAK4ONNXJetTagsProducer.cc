#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/ParticleTransformerAK4TagInfo.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

using namespace cms::Ort;

class ParticleTransformerAK4ONNXJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit ParticleTransformerAK4ONNXJetTagsProducer(const edm::ParameterSet&, const ONNXRuntime*);
  ~ParticleTransformerAK4ONNXJetTagsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

private:
  typedef std::vector<reco::ParticleTransformerAK4TagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  void produce(edm::Event&, const edm::EventSetup&) override;

  void make_inputs(unsigned i_jet, const reco::ParticleTransformerAK4TagInfo& taginfo);
  void get_input_sizes(edm::Handle<TagInfoCollection> tag_infos);

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  enum InputIndexes {
    kChargedCandidates = 0,
    kNeutralCandidates = 1,
    kVertices = 2,
    kChargedCandidates4Vec = 3,
    kNeutralCandidates4Vec = 4,
    kVertices4Vec = 5
  };
  unsigned n_cpf_;
  constexpr static unsigned n_features_cpf_ = 16;
  constexpr static unsigned n_pairwise_features_cpf_ = 4;
  unsigned n_npf_;
  constexpr static unsigned n_features_npf_ = 8;
  constexpr static unsigned n_pairwise_features_npf_ = 4;
  unsigned n_sv_;
  constexpr static unsigned n_features_sv_ = 14;
  constexpr static unsigned n_pairwise_features_sv_ = 4;
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

ParticleTransformerAK4ONNXJetTagsProducer::~ParticleTransformerAK4ONNXJetTagsProducer() {}

void ParticleTransformerAK4ONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfParticleTransformerAK4JetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfParticleTransformerAK4TagInfos"));
  desc.add<std::vector<std::string>>("input_names", {"input_1", "input_2", "input_3", "input_4", "input_5", "input_6"});
  desc.add<edm::FileInPath>("model_path",
                            edm::FileInPath("RecoBTag/Combined/data/RobustParTAK4/PUPPI/V00/ParTAK4.onnx"));
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

  std::vector<std::unique_ptr<JetTagCollection>> output_tags;
  if (!tag_infos->empty()) {
    // initialize output collection
    auto jet_ref = tag_infos->begin()->jet();
    auto ref2prod = edm::makeRefToBaseProdFrom(jet_ref, iEvent);
    for (std::size_t i = 0; i < flav_names_.size(); i++) {
      output_tags.emplace_back(std::make_unique<JetTagCollection>(ref2prod));
    }
    get_input_sizes(tag_infos);

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
    // run prediction with dynamic batch size per event
    input_shapes_ = {{(int64_t)tag_infos->size(), (int64_t)n_cpf_, (int64_t)n_features_cpf_},
                     {(int64_t)tag_infos->size(), (int64_t)n_npf_, (int64_t)n_features_npf_},
                     {(int64_t)tag_infos->size(), (int64_t)n_sv_, (int64_t)n_features_sv_},
                     {(int64_t)tag_infos->size(), (int64_t)n_cpf_, (int64_t)n_pairwise_features_cpf_},
                     {(int64_t)tag_infos->size(), (int64_t)n_npf_, (int64_t)n_pairwise_features_npf_},
                     {(int64_t)tag_infos->size(), (int64_t)n_sv_, (int64_t)n_pairwise_features_sv_}};

    auto outputs = globalCache()->run(input_names_, data_, input_shapes_, output_names_, tag_infos->size())[0];
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

void ParticleTransformerAK4ONNXJetTagsProducer::get_input_sizes(edm::Handle<TagInfoCollection> tag_infos) {
  for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
    const auto& taginfo = (*tag_infos)[jet_n];
    const auto& features = taginfo.features();
    unsigned int n_cpf = features.c_pf_features.size();
    unsigned int n_npf = features.n_pf_features.size();
    unsigned int n_vtx = features.sv_features.size();

    if (jet_n == 0) {
      n_cpf_ = std::max((unsigned int)1, n_cpf);
      n_npf_ = std::max((unsigned int)1, n_npf);
      n_sv_ = std::max((unsigned int)1, n_vtx);
    } else {
      n_cpf_ = std::max(n_cpf_, n_cpf);
      n_npf_ = std::max(n_npf_, n_npf);
      n_sv_ = std::max(n_sv_, n_vtx);
    }
  }
  n_cpf_ = std::min((unsigned int)25, n_cpf_);
  n_npf_ = std::min((unsigned int)25, n_npf_);
  n_sv_ = std::min((unsigned int)5, n_sv_);
  input_sizes_ = {
      n_cpf_ * n_features_cpf_,
      n_npf_ * n_features_npf_,
      n_sv_ * n_features_sv_,
      n_cpf_ * n_pairwise_features_cpf_,
      n_npf_ * n_pairwise_features_npf_,
      n_sv_ * n_pairwise_features_sv_,
  };
}

void ParticleTransformerAK4ONNXJetTagsProducer::make_inputs(unsigned i_jet,
                                                            const reco::ParticleTransformerAK4TagInfo& taginfo) {
  const auto& features = taginfo.features();
  float* ptr = nullptr;
  const float* start = nullptr;
  unsigned offset = 0;

  // c_pf candidates
  auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)n_cpf_);
  offset = i_jet * input_sizes_[kChargedCandidates];
  for (std::size_t c_pf_n = 0; c_pf_n < max_c_pf_n; c_pf_n++) {
    const auto& c_pf_features = features.c_pf_features.at(c_pf_n);
    ptr = &data_[kChargedCandidates][offset + c_pf_n * n_features_cpf_];
    start = ptr;
    *ptr = c_pf_features.btagPf_trackEtaRel;
    *(++ptr) = c_pf_features.btagPf_trackPtRel;
    *(++ptr) = c_pf_features.btagPf_trackPPar;
    *(++ptr) = c_pf_features.btagPf_trackDeltaR;
    *(++ptr) = c_pf_features.btagPf_trackPParRatio;
    *(++ptr) = c_pf_features.btagPf_trackSip2dVal;
    *(++ptr) = c_pf_features.btagPf_trackSip2dSig;
    *(++ptr) = c_pf_features.btagPf_trackSip3dVal;
    *(++ptr) = c_pf_features.btagPf_trackSip3dSig;
    *(++ptr) = c_pf_features.btagPf_trackJetDistVal;
    *(++ptr) = c_pf_features.ptrel;
    *(++ptr) = c_pf_features.drminsv;
    //*(++ptr) = c_pf_features.distminsv; // later during Run 3 after feature engineering
    *(++ptr) = c_pf_features.vtx_ass;
    *(++ptr) = c_pf_features.puppiw;
    *(++ptr) = c_pf_features.chi2;
    *(++ptr) = c_pf_features.quality;
    assert(start + n_features_cpf_ - 1 == ptr);
  }

  // n_pf candidates
  auto max_n_pf_n = std::min(features.n_pf_features.size(), (std::size_t)n_npf_);
  offset = i_jet * input_sizes_[kNeutralCandidates];
  for (std::size_t n_pf_n = 0; n_pf_n < max_n_pf_n; n_pf_n++) {
    const auto& n_pf_features = features.n_pf_features.at(n_pf_n);
    ptr = &data_[kNeutralCandidates][offset + n_pf_n * n_features_npf_];
    start = ptr;
    *ptr = n_pf_features.ptrel;
    *(++ptr) = n_pf_features.etarel;
    *(++ptr) = n_pf_features.phirel;
    *(++ptr) = n_pf_features.deltaR;
    *(++ptr) = n_pf_features.isGamma;
    *(++ptr) = n_pf_features.hadFrac;
    *(++ptr) = n_pf_features.drminsv;
    *(++ptr) = n_pf_features.puppiw;
    assert(start + n_features_npf_ - 1 == ptr);
  }

  // sv candidates
  auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)n_sv_);
  offset = i_jet * input_sizes_[kVertices];
  for (std::size_t sv_n = 0; sv_n < max_sv_n; sv_n++) {
    const auto& sv_features = features.sv_features.at(sv_n);
    ptr = &data_[kVertices][offset + sv_n * n_features_sv_];
    start = ptr;
    *ptr = sv_features.pt;
    *(++ptr) = sv_features.deltaR;
    *(++ptr) = sv_features.mass;
    *(++ptr) = sv_features.etarel;
    *(++ptr) = sv_features.phirel;
    *(++ptr) = sv_features.ntracks;
    *(++ptr) = sv_features.chi2;
    *(++ptr) = sv_features.normchi2;
    *(++ptr) = sv_features.dxy;
    *(++ptr) = sv_features.dxysig;
    *(++ptr) = sv_features.d3d;
    *(++ptr) = sv_features.d3dsig;
    *(++ptr) = sv_features.costhetasvpv;
    *(++ptr) = sv_features.enratio;
    assert(start + n_features_sv_ - 1 == ptr);
  }

  // cpf pairwise features (4-vectors)
  auto max_cpf_n = std::min(features.c_pf_features.size(), (std::size_t)n_cpf_);
  offset = i_jet * input_sizes_[kChargedCandidates4Vec];
  for (std::size_t cpf_n = 0; cpf_n < max_cpf_n; cpf_n++) {
    const auto& cpf_pairwise_features = features.c_pf_features.at(cpf_n);
    ptr = &data_[kChargedCandidates4Vec][offset + cpf_n * n_pairwise_features_cpf_];
    start = ptr;
    *ptr = cpf_pairwise_features.px;
    *(++ptr) = cpf_pairwise_features.py;
    *(++ptr) = cpf_pairwise_features.pz;
    *(++ptr) = cpf_pairwise_features.e;

    assert(start + n_pairwise_features_cpf_ - 1 == ptr);
  }

  // npf pairwise features (4-vectors)
  auto max_npf_n = std::min(features.n_pf_features.size(), (std::size_t)n_npf_);
  offset = i_jet * input_sizes_[kNeutralCandidates4Vec];
  for (std::size_t npf_n = 0; npf_n < max_npf_n; npf_n++) {
    const auto& npf_pairwise_features = features.n_pf_features.at(npf_n);
    ptr = &data_[kNeutralCandidates4Vec][offset + npf_n * n_pairwise_features_npf_];
    start = ptr;
    *ptr = npf_pairwise_features.px;
    *(++ptr) = npf_pairwise_features.py;
    *(++ptr) = npf_pairwise_features.pz;
    *(++ptr) = npf_pairwise_features.e;

    assert(start + n_pairwise_features_npf_ - 1 == ptr);
  }

  // sv pairwise features (4-vectors)
  auto max_sv_N = std::min(features.sv_features.size(), (std::size_t)n_sv_);
  offset = i_jet * input_sizes_[kVertices4Vec];
  for (std::size_t sv_N = 0; sv_N < max_sv_N; sv_N++) {
    const auto& sv_pairwise_features = features.sv_features.at(sv_N);
    ptr = &data_[kVertices4Vec][offset + sv_N * n_pairwise_features_sv_];
    start = ptr;
    *ptr = sv_pairwise_features.px;
    *(++ptr) = sv_pairwise_features.py;
    *(++ptr) = sv_pairwise_features.pz;
    *(++ptr) = sv_pairwise_features.e;

    assert(start + n_pairwise_features_sv_ - 1 == ptr);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ParticleTransformerAK4ONNXJetTagsProducer);
