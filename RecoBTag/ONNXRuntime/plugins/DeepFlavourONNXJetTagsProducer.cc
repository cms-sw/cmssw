#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"

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
  // get output names from flav_names
  for (const auto& flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }

  assert(input_names_.size() == input_sizes_.size());
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

  descriptions.add("pfDeepFlavourJetTags", desc);
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
  float* ptr = nullptr;
  const float* start = nullptr;
  unsigned offset = 0;

  // jet and other global features
  offset = i_jet * input_sizes_[kGlobal];
  ptr = &data_[kGlobal][offset];
  // jet variables
  const auto& jet_features = features.jet_features;
  start = ptr;
  *ptr = jet_features.pt;
  *(++ptr) = jet_features.eta;
  // number of elements in different collections
  *(++ptr) = features.c_pf_features.size();
  *(++ptr) = features.n_pf_features.size();
  *(++ptr) = features.sv_features.size();
  *(++ptr) = features.npv;
  // variables from ShallowTagInfo
  const auto& tag_info_features = features.tag_info_features;
  *(++ptr) = tag_info_features.trackSumJetEtRatio;
  *(++ptr) = tag_info_features.trackSumJetDeltaR;
  *(++ptr) = tag_info_features.vertexCategory;
  *(++ptr) = tag_info_features.trackSip2dValAboveCharm;
  *(++ptr) = tag_info_features.trackSip2dSigAboveCharm;
  *(++ptr) = tag_info_features.trackSip3dValAboveCharm;
  *(++ptr) = tag_info_features.trackSip3dSigAboveCharm;
  *(++ptr) = tag_info_features.jetNSelectedTracks;
  *(++ptr) = tag_info_features.jetNTracksEtaRel;
  assert(start + n_features_global_ - 1 == ptr);

  // c_pf candidates
  auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)25);
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
    *(++ptr) = c_pf_features.vtx_ass;
    *(++ptr) = c_pf_features.puppiw;
    *(++ptr) = c_pf_features.chi2;
    *(++ptr) = c_pf_features.quality;
    assert(start + n_features_cpf_ - 1 == ptr);
  }

  // n_pf candidates
  auto max_n_pf_n = std::min(features.n_pf_features.size(), (std::size_t)25);
  offset = i_jet * input_sizes_[kNeutralCandidates];
  for (std::size_t n_pf_n = 0; n_pf_n < max_n_pf_n; n_pf_n++) {
    const auto& n_pf_features = features.n_pf_features.at(n_pf_n);
    ptr = &data_[kNeutralCandidates][offset + n_pf_n * n_features_npf_];
    start = ptr;
    *ptr = n_pf_features.ptrel;
    *(++ptr) = n_pf_features.deltaR;
    *(++ptr) = n_pf_features.isGamma;
    *(++ptr) = n_pf_features.hadFrac;
    *(++ptr) = n_pf_features.drminsv;
    *(++ptr) = n_pf_features.puppiw;
    assert(start + n_features_npf_ - 1 == ptr);
  }

  // sv candidates
  auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)4);
  offset = i_jet * input_sizes_[kVertices];
  for (std::size_t sv_n = 0; sv_n < max_sv_n; sv_n++) {
    const auto& sv_features = features.sv_features.at(sv_n);
    ptr = &data_[kVertices][offset + sv_n * n_features_sv_];
    start = ptr;
    *ptr = sv_features.pt;
    *(++ptr) = sv_features.deltaR;
    *(++ptr) = sv_features.mass;
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

  // last input: jet pt
  offset = i_jet * input_sizes_[kJetPt];
  data_[kJetPt][offset] = features.jet_features.pt;
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepFlavourONNXJetTagsProducer);
