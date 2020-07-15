#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/DeepDoubleXTagInfo.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include <algorithm>

using namespace cms::Ort;

class DeepDoubleXONNXJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit DeepDoubleXONNXJetTagsProducer(const edm::ParameterSet&, const ONNXRuntime*);
  ~DeepDoubleXONNXJetTagsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

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
  const static std::vector<unsigned> input_sizes_;

  // hold the input data
  FloatArrays data_;
};

const std::vector<unsigned> DeepDoubleXONNXJetTagsProducer::input_sizes_{
    n_features_global_, n_cpf_* n_features_cpf_, n_sv_* n_features_sv_};

DeepDoubleXONNXJetTagsProducer::DeepDoubleXONNXJetTagsProducer(const edm::ParameterSet& iConfig,
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

DeepDoubleXONNXJetTagsProducer::~DeepDoubleXONNXJetTagsProducer() {}

void DeepDoubleXONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfDeepDoubleBvLJetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfDeepDoubleXTagInfos"));
  desc.add<std::vector<std::string>>("input_names", {"input_1", "input_2", "input_3"});
  desc.add<std::vector<std::string>>("output_names", {});

  using FIP = edm::FileInPath;
  using PDFIP = edm::ParameterDescription<edm::FileInPath>;
  using PDPSD = edm::ParameterDescription<std::vector<std::string>>;
  using PDCases = edm::ParameterDescriptionCases<std::string>;
  auto flavorCases = [&]() {
    return "BvL" >> (PDPSD("flav_names", std::vector<std::string>{"probQCD", "probHbb"}, true) and
                     PDFIP("model_path", FIP("RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDB.onnx"), true)) or
           "CvL" >> (PDPSD("flav_names", std::vector<std::string>{"probQCD", "probHcc"}, true) and
                     PDFIP("model_path", FIP("RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDC.onnx"), true)) or
           "CvB" >> (PDPSD("flav_names", std::vector<std::string>{"probHbb", "probHcc"}, true) and
                     PDFIP("model_path", FIP("RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDCvB.onnx"), true));
  };
  auto descBvL(desc);
  descBvL.ifValue(edm::ParameterDescription<std::string>("flavor", "BvL", true), flavorCases());
  descriptions.add("pfDeepDoubleBvLJetTags", descBvL);

  auto descCvL(desc);
  descCvL.ifValue(edm::ParameterDescription<std::string>("flavor", "CvL", true), flavorCases());
  descriptions.add("pfDeepDoubleCvLJetTags", descCvL);

  auto descCvB(desc);
  descCvB.ifValue(edm::ParameterDescription<std::string>("flavor", "CvB", true), flavorCases());
  descriptions.add("pfDeepDoubleCvBJetTags", descCvB);
}

std::unique_ptr<ONNXRuntime> DeepDoubleXONNXJetTagsProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void DeepDoubleXONNXJetTagsProducer::globalEndJob(const ONNXRuntime* cache) {}

void DeepDoubleXONNXJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

    std::vector<float> outputs;
    if (batch_size > 0) {
      // init data storage
      data_.clear();
      for (const auto& len : input_sizes_) {
        data_.emplace_back(batch_size * len, 0);
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
      outputs = globalCache()->run(input_names_, data_, output_names_, batch_size)[0];
      assert(outputs.size() == flav_names_.size() * batch_size);
    }

    // get the outputs
    unsigned i_output = 0;
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      const auto& taginfo = tag_infos->at(jet_n);
      const auto& jet_ref = taginfo.jet();
      for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
        if (!taginfo.features().empty()) {
          (*(output_tags[flav_n]))[jet_ref] = outputs[i_output];
          ++i_output;
        } else {
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

void DeepDoubleXONNXJetTagsProducer::make_inputs(unsigned i_jet, const reco::DeepDoubleXTagInfo& taginfo) {
  const auto& features = taginfo.features();
  float* ptr = nullptr;
  const float* start = nullptr;
  unsigned offset = 0;

  // DoubleB features
  offset = i_jet * input_sizes_[kGlobal];
  ptr = &data_[kGlobal][offset];
  start = ptr;
  const auto& tag_info_features = features.tag_info_features;
  *ptr = tag_info_features.jetNTracks;
  *(++ptr) = tag_info_features.jetNSecondaryVertices;
  *(++ptr) = tag_info_features.tau1_trackEtaRel_0;
  *(++ptr) = tag_info_features.tau1_trackEtaRel_1;
  *(++ptr) = tag_info_features.tau1_trackEtaRel_2;
  *(++ptr) = tag_info_features.tau2_trackEtaRel_0;
  *(++ptr) = tag_info_features.tau2_trackEtaRel_1;
  *(++ptr) = tag_info_features.tau2_trackEtaRel_2;
  *(++ptr) = tag_info_features.tau1_flightDistance2dSig;
  *(++ptr) = tag_info_features.tau2_flightDistance2dSig;
  *(++ptr) = tag_info_features.tau1_vertexDeltaR;
  // Note: this variable is not used in the 27-input BDT
  //    *(++ptr) = tag_info_features.tau2_vertexDeltaR;
  *(++ptr) = tag_info_features.tau1_vertexEnergyRatio;
  *(++ptr) = tag_info_features.tau2_vertexEnergyRatio;
  *(++ptr) = tag_info_features.tau1_vertexMass;
  *(++ptr) = tag_info_features.tau2_vertexMass;
  *(++ptr) = tag_info_features.trackSip2dSigAboveBottom_0;
  *(++ptr) = tag_info_features.trackSip2dSigAboveBottom_1;
  *(++ptr) = tag_info_features.trackSip2dSigAboveCharm;
  *(++ptr) = tag_info_features.trackSip3dSig_0;
  *(++ptr) = tag_info_features.tau1_trackSip3dSig_0;
  *(++ptr) = tag_info_features.tau1_trackSip3dSig_1;
  *(++ptr) = tag_info_features.trackSip3dSig_1;
  *(++ptr) = tag_info_features.tau2_trackSip3dSig_0;
  *(++ptr) = tag_info_features.tau2_trackSip3dSig_1;
  *(++ptr) = tag_info_features.trackSip3dSig_2;
  *(++ptr) = tag_info_features.trackSip3dSig_3;
  *(++ptr) = tag_info_features.z_ratio;
  assert(start + n_features_global_ - 1 == ptr);

  // c_pf candidates
  auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)n_cpf_);
  offset = i_jet * input_sizes_[kChargedCandidates];
  for (std::size_t c_pf_n = 0; c_pf_n < max_c_pf_n; c_pf_n++) {
    const auto& c_pf_features = features.c_pf_features.at(c_pf_n);
    ptr = &data_[kChargedCandidates][offset + c_pf_n * n_features_cpf_];
    start = ptr;
    *ptr = c_pf_features.btagPf_trackEtaRel;
    *(++ptr) = c_pf_features.btagPf_trackPtRatio;
    *(++ptr) = c_pf_features.btagPf_trackPParRatio;
    *(++ptr) = c_pf_features.btagPf_trackSip2dVal;
    *(++ptr) = c_pf_features.btagPf_trackSip2dSig;
    *(++ptr) = c_pf_features.btagPf_trackSip3dVal;
    *(++ptr) = c_pf_features.btagPf_trackSip3dSig;
    *(++ptr) = c_pf_features.btagPf_trackJetDistVal;
    assert(start + n_features_cpf_ - 1 == ptr);
  }

  // sv candidates
  auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)n_sv_);
  offset = i_jet * input_sizes_[kVertices];
  for (std::size_t sv_n = 0; sv_n < max_sv_n; sv_n++) {
    const auto& sv_features = features.sv_features.at(sv_n);
    ptr = &data_[kVertices][offset + sv_n * n_features_sv_];
    start = ptr;
    *ptr = sv_features.d3d;
    *(++ptr) = sv_features.d3dsig;
    assert(start + n_features_sv_ - 1 == ptr);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepDoubleXONNXJetTagsProducer);
