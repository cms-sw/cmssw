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

using namespace Ort;

class DeepJetTagsONNXProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit DeepJetTagsONNXProducer(const edm::ParameterSet&, const ONNXRuntime*);
  ~DeepJetTagsONNXProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

  enum InputIndexes { kGlobal = 0, kChargedCandidates = 1, kNeutralCandidates = 2, kVertices = 3, kJetPt = 4 };

private:
  typedef std::vector<reco::DeepFlavourTagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  void produce(edm::Event&, const edm::EventSetup&) override;

  void make_inputs(unsigned i_jet, const reco::DeepFlavourTagInfo& taginfo);

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;
  std::vector<std::string> input_names_;
  std::vector<unsigned int> input_sizes_;
  std::vector<std::string> output_names_;

  // hold the input data
  FloatArrays data_;
};

DeepJetTagsONNXProducer::DeepJetTagsONNXProducer(const edm::ParameterSet& iConfig, const ONNXRuntime* cache)
    : src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
      input_sizes_({15, 400, 150, 48, 1}),
      output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")) {
  // get output names from flav_names
  for (const auto& flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }

  assert(input_names_.size() == input_sizes_.size());
}

DeepJetTagsONNXProducer::~DeepJetTagsONNXProducer() {}

void DeepJetTagsONNXProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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

std::unique_ptr<ONNXRuntime> DeepJetTagsONNXProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void DeepJetTagsONNXProducer::globalEndJob(const ONNXRuntime* cache) {}

void DeepJetTagsONNXProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

  // put into the event
  for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
    iEvent.put(std::move(output_tags[flav_n]), flav_names_[flav_n]);
  }
}

void DeepJetTagsONNXProducer::make_inputs(unsigned i_jet, const reco::DeepFlavourTagInfo& taginfo) {
  const auto& features = taginfo.features();
  float* ptr = nullptr;
  unsigned offset = 0;

  // jet and other global features
  offset = i_jet * input_sizes_[kGlobal];
  ptr = &data_[0][offset];
  // jet variables
  const auto& jet_features = features.jet_features;
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

  // c_pf candidates
  auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)25);
  offset = i_jet * input_sizes_[kChargedCandidates];
  for (std::size_t c_pf_n = 0; c_pf_n < max_c_pf_n; c_pf_n++) {
    const auto& c_pf_features = features.c_pf_features.at(c_pf_n);
    ptr = &data_[1][offset + c_pf_n * 16];
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
  }

  // n_pf candidates
  auto max_n_pf_n = std::min(features.n_pf_features.size(), (std::size_t)25);
  offset = i_jet * input_sizes_[kNeutralCandidates];
  for (std::size_t n_pf_n = 0; n_pf_n < max_n_pf_n; n_pf_n++) {
    const auto& n_pf_features = features.n_pf_features.at(n_pf_n);
    ptr = &data_[2][offset + n_pf_n * 6];
    *ptr = n_pf_features.ptrel;
    *(++ptr) = n_pf_features.deltaR;
    *(++ptr) = n_pf_features.isGamma;
    *(++ptr) = n_pf_features.hadFrac;
    *(++ptr) = n_pf_features.drminsv;
    *(++ptr) = n_pf_features.puppiw;
  }

  // sv candidates
  auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)4);
  offset = i_jet * input_sizes_[kVertices];
  for (std::size_t sv_n = 0; sv_n < max_sv_n; sv_n++) {
    const auto& sv_features = features.sv_features.at(sv_n);
    ptr = &data_[3][offset + sv_n * 12];
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
  }

  // last input: jet pt
  offset = i_jet * input_sizes_[kJetPt];
  data_[4][offset] = features.jet_features.pt;
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepJetTagsONNXProducer);
