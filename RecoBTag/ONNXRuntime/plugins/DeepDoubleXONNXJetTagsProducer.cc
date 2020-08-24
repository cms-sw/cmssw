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
#include <iostream>
#include <fstream>

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
  std::string version_;

  unsigned n_features_global_;
  unsigned n_npf_, n_features_npf_;
  unsigned n_cpf_, n_features_cpf_;
  unsigned n_sv_, n_features_sv_;
  unsigned kGlobal, kNeutralCandidates, kChargedCandidates, kVertices;
  std::vector<unsigned> input_sizes_;

  // hold the input data
  FloatArrays data_;

  bool debug_ = false;
};

DeepDoubleXONNXJetTagsProducer::DeepDoubleXONNXJetTagsProducer(const edm::ParameterSet& iConfig,
                                                               const ONNXRuntime* cache)
    : src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
      output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")),
      version_(iConfig.getParameter<std::string>("version")),
      debug_(iConfig.getUntrackedParameter<bool>("debugMode", false)) {
  // get output names from flav_names
  for (const auto& flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }

  if (version_ == "V2") {
    n_features_global_ = 5;
    n_npf_ = 60;
    n_features_npf_ = 8;
    n_cpf_ = 40;
    n_features_cpf_ = 21;
    n_sv_ = 5;
    n_features_sv_ = 7;
    input_sizes_ = {n_features_global_, n_npf_ * n_features_npf_, n_cpf_ * n_features_cpf_, n_sv_ * n_features_sv_};
    kGlobal = 0;
    kNeutralCandidates = 1;
    kChargedCandidates = 2;
    kVertices = 3;
  } else {
    n_features_global_ = 27;
    n_cpf_ = 60;
    n_features_cpf_ = 8;
    n_sv_ = 5;
    n_features_sv_ = 2;
    input_sizes_ = {n_features_global_, n_cpf_ * n_features_cpf_, n_sv_ * n_features_sv_};
    kGlobal = 0;
    kChargedCandidates = 1;
    kVertices = 2;
  }

  assert(input_names_.size() == input_sizes_.size());
}

DeepDoubleXONNXJetTagsProducer::~DeepDoubleXONNXJetTagsProducer() {}

void DeepDoubleXONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfDeepDoubleXTagInfos"));
  desc.add<std::vector<std::string>>("input_names", {"input_1", "input_2", "input_3"});
  desc.add<std::vector<std::string>>("output_names", {});
  desc.add<std::string>("version", "V1");

  using FIP = edm::FileInPath;
  using PDFIP = edm::ParameterDescription<edm::FileInPath>;
  using PDPSD = edm::ParameterDescription<std::vector<std::string>>;
  using PDCases = edm::ParameterDescriptionCases<std::string>;
  using PDVersion = edm::ParameterDescription<std::string>;
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

    std::vector<float> etas_debug;
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
          if (debug_) {
            etas_debug.push_back(taginfo.jet()->eta());
          }
          ++idx;
        }
      }

      std::sort(input_names_.begin(), input_names_.end());  // input_names order on input is not preserved
      // run prediction
      outputs = globalCache()->run(input_names_, data_, {}, output_names_, batch_size)[0];

      if (debug_) {
        // Dump inputs to file
        std::ofstream outfile;
        outfile.open("test.txt", std::ios_base::app);
        outfile << iEvent.id().event() << std::endl;
        outfile << batch_size << std::endl;
        for (float x : etas_debug)
          outfile << x << ' ';
        outfile << std::endl;
        int _i = 0;
        for (const std::vector<float>& v : data_) {
          outfile << "input_" << _i << std::endl;
          for (float x : v)
            outfile << x << ' ';
          outfile << std::endl;
          _i = _i + 1;
        }
        outfile << "outputs" << std::endl;
        for (float x : outputs)
          outfile << x << ' ';
        outfile << std::endl;
      }

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
  if (version_ == "V1") {
    *(++ptr) = tag_info_features.jetNSecondaryVertices;
  }
  *(++ptr) = tag_info_features.tau1_trackEtaRel_0;
  *(++ptr) = tag_info_features.tau1_trackEtaRel_1;
  if (version_ == "V1") {
    *(++ptr) = tag_info_features.tau1_trackEtaRel_2;
    *(++ptr) = tag_info_features.tau2_trackEtaRel_0;
    *(++ptr) = tag_info_features.tau2_trackEtaRel_1;
    *(++ptr) = tag_info_features.tau2_trackEtaRel_2;
    *(++ptr) = tag_info_features.tau1_flightDistance2dSig;
    *(++ptr) = tag_info_features.tau2_flightDistance2dSig;
  }
  *(++ptr) = tag_info_features.tau1_vertexDeltaR;
  *(++ptr) = tag_info_features.tau1_vertexEnergyRatio;
  if (version_ == "V1") {
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
  }
  assert(start + n_features_global_ - 1 == ptr);

  // c_pf candidates
  auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)n_cpf_);
  offset = i_jet * input_sizes_[kChargedCandidates];
  for (std::size_t c_pf_n = 0; c_pf_n < max_c_pf_n; c_pf_n++) {
    const auto& c_pf_features = features.c_pf_features.at(c_pf_n);
    ptr = &data_[kChargedCandidates][offset + c_pf_n * n_features_cpf_];
    start = ptr;
    *ptr = c_pf_features.btagPf_trackEtaRel;
    if (version_ == "V1") {
      *(++ptr) = c_pf_features.btagPf_trackPtRatio;
      *(++ptr) = c_pf_features.btagPf_trackPParRatio;
      *(++ptr) = c_pf_features.btagPf_trackSip2dVal;
      *(++ptr) = c_pf_features.btagPf_trackSip2dSig;
      *(++ptr) = c_pf_features.btagPf_trackSip3dVal;
      *(++ptr) = c_pf_features.btagPf_trackSip3dSig;
      *(++ptr) = c_pf_features.btagPf_trackJetDistVal;
    } else {
      *(++ptr) = c_pf_features.btagPf_trackJetDistVal;
      *(++ptr) = c_pf_features.btagPf_trackPParRatio;
      *(++ptr) = c_pf_features.btagPf_trackPtRatio;
      *(++ptr) = c_pf_features.btagPf_trackSip2dSig;
      *(++ptr) = c_pf_features.btagPf_trackSip2dVal;
      *(++ptr) = c_pf_features.btagPf_trackSip3dSig;
      *(++ptr) = c_pf_features.btagPf_trackSip3dVal;
      *(++ptr) = c_pf_features.deltaR;
      *(++ptr) = c_pf_features.drminsv;
      *(++ptr) = c_pf_features.drsubjet1;
      *(++ptr) = c_pf_features.drsubjet2;
      *(++ptr) = c_pf_features.dxy;
      *(++ptr) = c_pf_features.dxysig;
      *(++ptr) = c_pf_features.dz;
      *(++ptr) = c_pf_features.dzsig;
      *(++ptr) = c_pf_features.erel;
      *(++ptr) = c_pf_features.etarel;
      *(++ptr) = c_pf_features.chi2;
      *(++ptr) = c_pf_features.ptrel_noclip;
      *(++ptr) = c_pf_features.quality;
    }

    assert(start + n_features_cpf_ - 1 == ptr);
  }

  if (version_ == "V2") {
    // n_pf candidates
    auto max_n_pf_n = std::min(features.n_pf_features.size(), (std::size_t)n_cpf_);
    offset = i_jet * input_sizes_[kNeutralCandidates];
    for (std::size_t n_pf_n = 0; n_pf_n < max_n_pf_n; n_pf_n++) {
      const auto& n_pf_features = features.n_pf_features.at(n_pf_n);
      ptr = &data_[kNeutralCandidates][offset + n_pf_n * n_features_npf_];
      start = ptr;
      *ptr = n_pf_features.deltaR_noclip;
      *(++ptr) = n_pf_features.drminsv;
      *(++ptr) = n_pf_features.drsubjet1;
      *(++ptr) = n_pf_features.drsubjet2;
      *(++ptr) = n_pf_features.erel;
      *(++ptr) = n_pf_features.hadFrac;
      *(++ptr) = n_pf_features.ptrel_noclip;
      *(++ptr) = n_pf_features.puppiw;
      assert(start + n_features_npf_ - 1 == ptr);
    }
  }

  // sv candidates
  auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)n_sv_);
  offset = i_jet * input_sizes_[kVertices];
  for (std::size_t sv_n = 0; sv_n < max_sv_n; sv_n++) {
    const auto& sv_features = features.sv_features.at(sv_n);
    ptr = &data_[kVertices][offset + sv_n * n_features_sv_];
    start = ptr;
    if (version_ == "V1") {
      *ptr = sv_features.d3d;
      *(++ptr) = sv_features.d3dsig;
    } else {
      *ptr = sv_features.costhetasvpv;
      *(++ptr) = sv_features.deltaR;
      *(++ptr) = sv_features.dxysig;
      *(++ptr) = sv_features.mass;
      *(++ptr) = sv_features.ntracks;
      *(++ptr) = sv_features.pt;
      *(++ptr) = sv_features.ptrel;
    }
    assert(start + n_features_sv_ - 1 == ptr);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepDoubleXONNXJetTagsProducer);
