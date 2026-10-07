#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/HLTParticleTransformerAK4Features.h"
#include "DataFormats/BTauReco/interface/HLTParticleTransformerAK4TagInfo.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

using namespace cms::Ort;

class HLTParticleTransformerAK4ONNXJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit HLTParticleTransformerAK4ONNXJetTagsProducer(const edm::ParameterSet&, const ONNXRuntime*);
  ~HLTParticleTransformerAK4ONNXJetTagsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

private:
  typedef std::vector<reco::HLTParticleTransformerAK4TagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  void produce(edm::Event&, const edm::EventSetup&) override;

  void make_inputs(const btagbtvdeep::HLTParticleTransformerAK4Features& features);
  void get_input_sizes(const reco::HLTParticleTransformerAK4TagInfo& taginfo);

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  enum InputIndexes { kGlobalFeatures = 0, kCpfCandidates = 1, kVtxFeatures = 2 };

  constexpr static size_t global_size_ = 4;
  constexpr static unsigned n_max_cpf_candidates_ = 50;
  constexpr static unsigned n_features_cpf_ = 28;
  constexpr static unsigned n_max_sv_candidates_ = 5;
  constexpr static unsigned n_features_sv_ = 16;

  std::vector<std::vector<int64_t>> input_shapes_;

  FloatArrays data_;
};

HLTParticleTransformerAK4ONNXJetTagsProducer::HLTParticleTransformerAK4ONNXJetTagsProducer(
    const edm::ParameterSet& iConfig, const ONNXRuntime* cache)
    : src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
      output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")) {
  for (const auto& flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }
}

void HLTParticleTransformerAK4ONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltParticleTransformerAK4TagInfos"));
  desc.add<std::vector<std::string>>("input_names", {"global_features", "cpf_features", "vtx_features"});
  desc.add<edm::FileInPath>(
      "model_path",
      edm::FileInPath("RecoBTag/Combined/data/HLT/hltParticleTransformerAK4/hltParTAK4_CMSSW15_082026.onnx"));
  desc.add<std::vector<std::string>>("output_names", {"output"});
  desc.add<std::vector<std::string>>("flav_names", {"probb", "probbb", "problepb"});

  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<ONNXRuntime> HLTParticleTransformerAK4ONNXJetTagsProducer::initializeGlobalCache(
    const edm::ParameterSet& iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void HLTParticleTransformerAK4ONNXJetTagsProducer::globalEndJob(const ONNXRuntime* cache) {}

void HLTParticleTransformerAK4ONNXJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(src_, tag_infos);

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

      input_shapes_ = {{1, (int64_t)global_size_},
                       {1, (int64_t)n_max_cpf_candidates_, (int64_t)n_features_cpf_},
                       {1, (int64_t)n_max_sv_candidates_, (int64_t)n_features_sv_}};

      outputs = globalCache()->run(input_names_, data_, input_shapes_, output_names_, 1)[0];
      assert(outputs.size() == flav_names_.size());
    }

    const auto& jet_ref = taginfo.jet();
    for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
      (*(output_tags[flav_n]))[jet_ref] = outputs[flav_n];
    }
  }

  for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
    iEvent.put(std::move(output_tags[flav_n]), flav_names_[flav_n]);
  }
}

void HLTParticleTransformerAK4ONNXJetTagsProducer::get_input_sizes(
    const reco::HLTParticleTransformerAK4TagInfo& taginfo) {
  const auto& features = taginfo.features();

  std::vector<unsigned int> input_sizes = {static_cast<unsigned int>(global_size_),
                                           static_cast<unsigned int>(n_max_cpf_candidates_ * n_features_cpf_),
                                           static_cast<unsigned int>(n_max_sv_candidates_ * n_features_sv_)};

  data_.clear();
  for (const auto& len : input_sizes) {
    data_.emplace_back(len, 0);
  }

  make_inputs(features);
}

void HLTParticleTransformerAK4ONNXJetTagsProducer::make_inputs(
    const btagbtvdeep::HLTParticleTransformerAK4Features& features) {
  float* ptr = nullptr;
  {
    float* start = &data_[kGlobalFeatures][0];
    ptr = start;
    *ptr++ = features.global_features.jet_pt;
    *ptr++ = features.global_features.jet_eta;
    *ptr++ = features.global_features.jet_phi;
    *ptr++ = features.global_features.jet_energy;
    assert(ptr == start + global_size_);
  }

  {
    for (std::size_t c_pf_n = 0; c_pf_n < std::min(features.cpf_candidates.size(), (std::size_t)n_max_cpf_candidates_);
         c_pf_n++) {
      ptr = &data_[kCpfCandidates][c_pf_n * n_features_cpf_];
      const auto& cpf = features.cpf_candidates[c_pf_n];
      float* start_cpf = ptr;

      *ptr++ = cpf.jet_pfcand_deta;
      *ptr++ = cpf.jet_pfcand_dphi;
      *ptr++ = cpf.jet_pfcand_pt_log;
      *ptr++ = cpf.jet_pfcand_energy_log;
      *ptr++ = cpf.jet_pfcand_charge;
      *ptr++ = cpf.jet_pfcand_frompv;
      *ptr++ = cpf.jet_pfcand_nlostinnerhits;
      *ptr++ = cpf.jet_pfcand_track_chi2;
      *ptr++ = cpf.jet_pfcand_track_qual;
      *ptr++ = cpf.jet_pfcand_dz;
      *ptr++ = cpf.jet_pfcand_dzsig;
      *ptr++ = cpf.jet_pfcand_dxy;
      *ptr++ = cpf.jet_pfcand_dxysig;
      *ptr++ = cpf.jet_pfcand_etarel;
      *ptr++ = cpf.jet_pfcand_pperp_ratio;
      *ptr++ = cpf.jet_pfcand_ppara_ratio;
      *ptr++ = cpf.jet_pfcand_trackjet_d3d;
      *ptr++ = cpf.jet_pfcand_trackjet_d3dsig;
      *ptr++ = cpf.jet_pfcand_trackjet_dist;
      *ptr++ = cpf.jet_pfcand_trackjet_decayL;
      *ptr++ = cpf.jet_pfcand_npixhits;
      *ptr++ = cpf.jet_pfcand_nstriphits;
      *ptr++ = cpf.jet_pfcand_highpurity;
      *ptr++ = cpf.jet_pfcand_id;
      *ptr++ = cpf.jet_pfcand_pt;
      *ptr++ = cpf.jet_pfcand_eta;
      *ptr++ = cpf.jet_pfcand_phi;
      *ptr++ = cpf.jet_pfcand_energy;

      assert(ptr - start_cpf == static_cast<int>(n_features_cpf_));
    }
  }

  {
    assert(data_[kVtxFeatures].size() >= n_max_sv_candidates_ * n_features_sv_);
    for (std::size_t sv_n = 0; sv_n < std::min(features.vtx_features.size(), (std::size_t)n_max_sv_candidates_);
         sv_n++) {
      ptr = &data_[kVtxFeatures][sv_n * n_features_sv_];
      const auto& sv = features.vtx_features[sv_n];
      float* start_sv = ptr;

      *ptr++ = sv.jet_sv_deta;
      *ptr++ = sv.jet_sv_dphi;
      *ptr++ = sv.jet_sv_pt_log;
      *ptr++ = sv.jet_sv_mass;
      *ptr++ = sv.jet_sv_ntrack;
      *ptr++ = sv.jet_sv_chi2;
      *ptr++ = sv.jet_sv_dxy;
      *ptr++ = sv.jet_sv_dxysig;
      *ptr++ = sv.jet_sv_d3d;
      *ptr++ = sv.jet_sv_d3dsig;
      *ptr++ = sv.jet_sv_costhetasvpv;
      *ptr++ = sv.jet_sv_enratio;
      *ptr++ = sv.jet_sv_pt;
      *ptr++ = sv.jet_sv_eta;
      *ptr++ = sv.jet_sv_phi;
      *ptr++ = sv.jet_sv_energy;

      assert(ptr - start_sv == static_cast<int>(n_features_sv_));
    }
  }
}

DEFINE_FWK_MODULE(HLTParticleTransformerAK4ONNXJetTagsProducer);
