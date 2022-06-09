#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/DeepBoostedJetTagInfo.h"

#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"

#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"

#include "RecoBTag/FeatureTools/interface/deep_helpers.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <nlohmann/json.hpp>

using namespace btagbtvdeep;

class ParticleNetSonicJetTagsProducer : public TritonEDProducer<> {
public:
  explicit ParticleNetSonicJetTagsProducer(const edm::ParameterSet &);
  ~ParticleNetSonicJetTagsProducer() override;

  void acquire(edm::Event const &iEvent, edm::EventSetup const &iSetup, Input &iInput) override;
  void produce(edm::Event &iEvent, edm::EventSetup const &iSetup, Output const &iOutput) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  typedef std::vector<reco::DeepBoostedJetTagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  void make_inputs(const reco::DeepBoostedJetTagInfo &taginfo);

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;             // names of the output scores
  std::vector<std::string> input_names_;            // names of each input group - the ordering is important!
  std::vector<std::vector<int64_t>> input_shapes_;  // shapes of each input group (-1 for dynamic axis)
  std::vector<unsigned> input_sizes_;               // total length of each input vector
  std::unordered_map<std::string, PreprocessParams> prep_info_map_;  // preprocessing info for each input group
  bool debug_ = false;
  bool skippedInference_ = false;
  constexpr static unsigned numParticleGroups_ = 3;
};

ParticleNetSonicJetTagsProducer::ParticleNetSonicJetTagsProducer(const edm::ParameterSet &iConfig)
    : TritonEDProducer<>(iConfig),
      src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      debug_(iConfig.getUntrackedParameter<bool>("debugMode", false)) {
  ParticleNetConstructor(iConfig, false, input_names_, prep_info_map_, input_shapes_, input_sizes_, nullptr);

  if (debug_) {
    LogDebug("ParticleNetSonicJetTagsProducer") << "<ParticleNetSonicJetTagsProducer::produce>:" << std::endl;
    for (unsigned i = 0; i < input_names_.size(); ++i) {
      const auto &group_name = input_names_.at(i);
      if (!input_shapes_.empty()) {
        LogDebug("ParticleNetSonicJetTagsProducer") << group_name << "\nshapes: ";
        for (const auto &x : input_shapes_.at(i)) {
          LogDebug("ParticleNetSonicJetTagsProducer") << x << ", ";
        }
      }
      LogDebug("ParticleNetSonicJetTagsProducer") << "\nvariables: ";
      for (const auto &x : prep_info_map_.at(group_name).var_names) {
        LogDebug("ParticleNetSonicJetTagsProducer") << x << ", ";
      }
      LogDebug("ParticleNetSonicJetTagsProducer") << "\n";
    }
    LogDebug("ParticleNetSonicJetTagsProducer") << "flav_names: ";
    for (const auto &flav_name : flav_names_) {
      LogDebug("ParticleNetSonicJetTagsProducer") << flav_name << ", ";
    }
    LogDebug("ParticleNetSonicJetTagsProducer") << "\n";
  }

  // get output names from flav_names
  for (const auto &flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }
}

ParticleNetSonicJetTagsProducer::~ParticleNetSonicJetTagsProducer() {}

void ParticleNetSonicJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // pfDeepBoostedJetTags
  edm::ParameterSetDescription desc;
  TritonClient::fillPSetDescription(desc);
  desc.add<edm::InputTag>("src", edm::InputTag("pfDeepBoostedJetTagInfos"));
  desc.add<std::string>("preprocess_json", "");
  // `preprocessParams` is deprecated -- use the preprocessing json file instead
  edm::ParameterSetDescription preprocessParams;
  preprocessParams.setAllowAnything();
  preprocessParams.setComment("`preprocessParams` is deprecated, please use `preprocess_json` instead.");
  desc.addOptional<edm::ParameterSetDescription>("preprocessParams", preprocessParams);
  desc.add<std::vector<std::string>>("flav_names",
                                     std::vector<std::string>{
                                         "probTbcq",
                                         "probTbqq",
                                         "probTbc",
                                         "probTbq",
                                         "probWcq",
                                         "probWqq",
                                         "probZbb",
                                         "probZcc",
                                         "probZqq",
                                         "probHbb",
                                         "probHcc",
                                         "probHqqqq",
                                         "probQCDbb",
                                         "probQCDcc",
                                         "probQCDb",
                                         "probQCDc",
                                         "probQCDothers",
                                     });
  desc.addOptionalUntracked<bool>("debugMode", false);

  descriptions.addWithDefaultLabel(desc);
}

void ParticleNetSonicJetTagsProducer::acquire(edm::Event const &iEvent, edm::EventSetup const &iSetup, Input &iInput) {
  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(src_, tag_infos);
  client_->setBatchSize(tag_infos->size());
  skippedInference_ = false;
  if (!tag_infos->empty()) {
    unsigned int maxParticles = 0;
    unsigned int maxVertices = 0;
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      maxParticles = std::max(maxParticles,
                              static_cast<unsigned int>(((*tag_infos)[jet_n]).features().get("pfcand_etarel").size()));
      maxVertices =
          std::max(maxVertices, static_cast<unsigned int>(((*tag_infos)[jet_n]).features().get("sv_etarel").size()));
    }
    if (maxParticles == 0 && maxVertices == 0) {
      client_->setBatchSize(0);
      skippedInference_ = true;
      return;
    }
    unsigned int minPartFromJSON = prep_info_map_.at(input_names_[0]).min_length;
    unsigned int maxPartFromJSON = prep_info_map_.at(input_names_[0]).max_length;
    unsigned int minVertFromJSON = prep_info_map_.at(input_names_[3]).min_length;
    unsigned int maxVertFromJSON = prep_info_map_.at(input_names_[3]).max_length;
    maxParticles = std::clamp(maxParticles, minPartFromJSON, maxPartFromJSON);
    maxVertices = std::clamp(maxVertices, minVertFromJSON, maxVertFromJSON);

    for (unsigned igroup = 0; igroup < input_names_.size(); ++igroup) {
      const auto &group_name = input_names_[igroup];
      auto &input = iInput.at(group_name);
      unsigned target;
      if (igroup < numParticleGroups_) {
        input.setShape(1, maxParticles);
        target = maxParticles;
      } else {
        input.setShape(1, maxVertices);
        target = maxVertices;
      }
      auto tdata = input.allocate<float>(true);
      for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
        const auto &taginfo = (*tag_infos)[jet_n];
        auto &vdata = (*tdata)[jet_n];
        const auto &prep_params = prep_info_map_.at(group_name);
        unsigned curr_pos = 0;
        // transform/pad
        for (unsigned i = 0; i < prep_params.var_names.size(); ++i) {
          const auto &varname = prep_params.var_names[i];
          const auto &raw_value = taginfo.features().get(varname);
          const auto &info = prep_params.info(varname);
          int insize = center_norm_pad_halfRagged(raw_value,
                                                  info.center,
                                                  info.norm_factor,
                                                  target,
                                                  vdata,
                                                  curr_pos,
                                                  info.pad,
                                                  info.replace_inf_value,
                                                  info.lower_bound,
                                                  info.upper_bound);
          curr_pos += insize;
          if (i == 0 && (!input_shapes_.empty())) {
            input_shapes_[igroup][2] = insize;
          }

          if (debug_) {
            LogDebug("acquire") << "<ParticleNetSonicJetTagsProducer::produce>:" << std::endl
                                << " -- var=" << varname << ", center=" << info.center << ", scale=" << info.norm_factor
                                << ", replace=" << info.replace_inf_value << ", pad=" << info.pad << std::endl;
            for (unsigned i = curr_pos - insize; i < curr_pos; i++) {
              LogDebug("acquire") << vdata[i] << ",";
            }
            LogDebug("acquire") << std::endl;
          }
        }
      }
      input.toServer(tdata);
    }
  }
}

void ParticleNetSonicJetTagsProducer::produce(edm::Event &iEvent,
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

        if (!taginfo.features().empty()) {
          for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
            (*(output_tags[flav_n]))[jet_ref] = outputs_from_server[jet_n][flav_n];
          }
        } else {
          for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
            (*(output_tags[flav_n]))[jet_ref] = 0.;
          }
        }
      }
    } else {
      for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
        const auto &jet_ref = tag_infos->at(jet_n).jet();
        for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
          (*(output_tags[flav_n]))[jet_ref] = 0.;
        }
      }
    }
  }

  if (debug_) {
    LogDebug("produce") << "<ParticleNetSonicJetTagsProducer::produce>:" << std::endl
                        << "=== " << iEvent.id().run() << ":" << iEvent.id().luminosityBlock() << ":"
                        << iEvent.id().event() << " ===" << std::endl;
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      const auto &jet_ref = tag_infos->at(jet_n).jet();
      LogDebug("produce") << " - Jet #" << jet_n << ", pt=" << jet_ref->pt() << ", eta=" << jet_ref->eta()
                          << ", phi=" << jet_ref->phi() << std::endl;
      for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
        LogDebug("produce") << "    " << flav_names_.at(flav_n) << " = " << (*(output_tags.at(flav_n)))[jet_ref]
                            << std::endl;
      }
    }
  }

  // put into the event
  for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
    iEvent.put(std::move(output_tags[flav_n]), flav_names_[flav_n]);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ParticleNetSonicJetTagsProducer);
