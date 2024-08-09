#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/ParticleTransformerAK4TagInfo.h"
#include "DataFormats/BTauReco/interface/ParticleTransformerAK4Features.h"

#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"

#include "RecoBTag/ONNXRuntime/interface/tensor_fillers.h"
#include "RecoBTag/ONNXRuntime/interface/tensor_configs.h"

class ParticleTransformerAK4SonicJetTagsProducer : public TritonEDProducer<> {
public:
  explicit ParticleTransformerAK4SonicJetTagsProducer(const edm::ParameterSet &);
  ~ParticleTransformerAK4SonicJetTagsProducer() override;

  void acquire(edm::Event const &iEvent, edm::EventSetup const &iSetup, Input &iInput) override;

  void produce(edm::Event &iEvent, edm::EventSetup const &iSetup, Output const &iOutput) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  typedef std::vector<reco::ParticleTransformerAK4TagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  bool skippedInference_ = false;
};

ParticleTransformerAK4SonicJetTagsProducer::ParticleTransformerAK4SonicJetTagsProducer(const edm::ParameterSet &iConfig)
    : TritonEDProducer<>(iConfig),
      src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
      output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")) {
  // get output names from flav_names
  for (const auto &flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }
}

ParticleTransformerAK4SonicJetTagsProducer::~ParticleTransformerAK4SonicJetTagsProducer() {}

void ParticleTransformerAK4SonicJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // pfParticleTransformerAK4JetTags
  edm::ParameterSetDescription desc;
  TritonClient::fillPSetDescription(desc);
  desc.add<edm::InputTag>("src", edm::InputTag("pfParticleTransformerAK4TagInfos"));
  desc.add<std::vector<std::string>>("input_names", {"input_1", "input_2", "input_3", "input_4", "input_5", "input_6"});
  desc.add<std::vector<std::string>>("output_names", {"softmax"});
  desc.add<std::vector<std::string>>(
      "flav_names", std::vector<std::string>{"probb", "probbb", "problepb", "probc", "probuds", "probg"});

  descriptions.add("pfParticleTransformerAK4SonicJetTags", desc);
}

void ParticleTransformerAK4SonicJetTagsProducer::acquire(edm::Event const &iEvent,
                                                         edm::EventSetup const &iSetup,
                                                         Input &iInput) {
  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(src_, tag_infos);
  client_->setBatchSize(tag_infos->size());
  skippedInference_ = false;
  if (tag_infos->empty())
    return;

  // Find the max n_cpf, n_npf and n_vtx among all the jets in an event.
  unsigned int max_n_cpf_counter = 0;
  unsigned int max_n_npf_counter = 0;
  unsigned int max_n_vtx_counter = 0;
  for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
    max_n_cpf_counter =
        std::max(max_n_cpf_counter, static_cast<unsigned int>(((*tag_infos)[jet_n]).features().c_pf_features.size()));
    max_n_npf_counter =
        std::max(max_n_npf_counter, static_cast<unsigned int>(((*tag_infos)[jet_n]).features().n_pf_features.size()));
    max_n_vtx_counter =
        std::max(max_n_vtx_counter, static_cast<unsigned int>(((*tag_infos)[jet_n]).features().sv_features.size()));
  }

  // If an event has no jet, or all jets has zero n_cpf, n_npf and n_vtx, the inference is skipped.
  if (max_n_cpf_counter == 0 && max_n_npf_counter == 0 && max_n_vtx_counter == 0) {
    client_->setBatchSize(0);
    skippedInference_ = true;
    return;
  }

  // all the jets in the same event will fill up the same amount of n_cpf, n_npf, n_vtx and send to server
  const unsigned int target_n_cpf = std::clamp(max_n_cpf_counter, (unsigned int)1, (unsigned int)parT::n_cpf_accept);
  const unsigned int target_n_npf = std::clamp(max_n_npf_counter, (unsigned int)1, (unsigned int)parT::n_npf_accept);
  const unsigned int target_n_vtx = std::clamp(max_n_vtx_counter, (unsigned int)1, (unsigned int)parT::n_sv_accept);

  const std::map<parT::InputFeatures, unsigned int> target_n{{parT::kChargedCandidates, target_n_cpf},
                                                             {parT::kNeutralCandidates, target_n_npf},
                                                             {parT::kVertices, target_n_vtx},
                                                             {parT::kChargedCandidates4Vec, target_n_cpf},
                                                             {parT::kNeutralCandidates4Vec, target_n_npf},
                                                             {parT::kVertices4Vec, target_n_vtx}};

  // loop through all groups of features
  for (parT::InputFeatures ifeature = parT::kBegin; ifeature != parT::kEnd;
       ifeature = static_cast<parT::InputFeatures>(ifeature + 1)) {
    const auto &group_name = input_names_[ifeature];
    auto &input = iInput.at(group_name);

    input.setShape(0, target_n.at(ifeature));
    auto tdata = input.allocate<float>(true);

    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      const auto &taginfo = (*tag_infos)[jet_n];
      const auto &features = taginfo.features();
      auto &vdata = (*tdata)[jet_n];
      if (ifeature == parT::kChargedCandidates || ifeature == parT::kChargedCandidates4Vec)
        parT_tensor_filler(vdata, ifeature, features.c_pf_features, target_n_cpf);
      else if (ifeature == parT::kNeutralCandidates || ifeature == parT::kNeutralCandidates4Vec)
        parT_tensor_filler(vdata, ifeature, features.n_pf_features, target_n_npf);
      else if (ifeature == parT::kVertices || ifeature == parT::kVertices4Vec)
        parT_tensor_filler(vdata, ifeature, features.sv_features, target_n_vtx);
    }
    input.toServer(tdata);
  }
}

void ParticleTransformerAK4SonicJetTagsProducer::produce(edm::Event &iEvent,
                                                         const edm::EventSetup &iSetup,
                                                         Output const &iOutput) {
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
  if (!tag_infos->empty()) {
    if (!skippedInference_) {
      const auto &output1 = iOutput.begin()->second;
      const auto &outputs_from_server = output1.fromServer<float>();

      for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
        const auto &taginfo = (*tag_infos)[jet_n];
        const auto &jet_ref = tag_infos->at(jet_n).jet();

        if (taginfo.features().is_filled) {
          for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
            (*(output_tags[flav_n]))[jet_ref] = outputs_from_server[jet_n][flav_n];
          }
        } else {
          for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
            (*(output_tags[flav_n]))[jet_ref] = -1.0;
          }
        }
      }
    } else {
      for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
        const auto &jet_ref = tag_infos->at(jet_n).jet();
        for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
          (*(output_tags[flav_n]))[jet_ref] = -1.0;
        }
      }
    }
  }
  // put into the event
  for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
    iEvent.put(std::move(output_tags[flav_n]), flav_names_[flav_n]);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ParticleTransformerAK4SonicJetTagsProducer);
