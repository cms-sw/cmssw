// -*- C++ -*-
//
// Package:    PhysicsTools/PatAlgos
// Class:      PNETLeptonProducer
//
// Original Author:  Sergio Sanchez Cruz
//         Created:  Mon, 15 May 2023 08:32:03 GMT
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/DeepBoostedJetFeatures.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoBTag/FeatureTools/interface/deep_helpers.h"

using namespace cms::Ort;
using namespace btagbtvdeep;

template <typename LeptonType>
class PNETLeptonProducer : public edm::stream::EDProducer<edm::GlobalCache<cms::Ort::ONNXRuntime>> {
public:
  explicit PNETLeptonProducer(const edm::ParameterSet &, const cms::Ort::ONNXRuntime *);
  ~PNETLeptonProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &);
  static void globalEndJob(const ONNXRuntime *);

private:
  using LeptonTagInfoCollection = DeepBoostedJetFeaturesCollection;

  void produce(edm::Event &, const edm::EventSetup &) override;
  void make_inputs(const DeepBoostedJetFeatures &);

  edm::EDGetTokenT<LeptonTagInfoCollection> src_;
  edm::EDGetTokenT<edm::View<LeptonType>> leps_;
  std::vector<std::string> flav_names_;
  std::vector<std::string> input_names_;            // names of each input group - the ordering is important!
  std::vector<std::vector<int64_t>> input_shapes_;  // shapes of each input group (-1 for dynamic axis)
  std::vector<unsigned> input_sizes_;               // total length of each input vector
  std::unordered_map<std::string, btagbtvdeep::PreprocessParams>
      prep_info_map_;  // preprocessing info for each input group

  cms::Ort::FloatArrays data_;
  bool debug_ = false;
};

template <typename LeptonType>
PNETLeptonProducer<LeptonType>::PNETLeptonProducer(const edm::ParameterSet &iConfig, const cms::Ort::ONNXRuntime *cache)
    : src_(consumes<LeptonTagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      leps_(consumes<edm::View<LeptonType>>(iConfig.getParameter<edm::InputTag>("srcLeps"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      debug_(iConfig.getUntrackedParameter<bool>("debugMode", false)) {
  ParticleNetConstructor(iConfig, true, input_names_, prep_info_map_, input_shapes_, input_sizes_, &data_);

  if (debug_) {
    for (unsigned i = 0; i < input_names_.size(); ++i) {
      const auto &group_name = input_names_.at(i);
      std::cout << group_name << std::endl;
      if (!input_shapes_.empty()) {
        std::cout << group_name << "\nshapes: ";
        for (const auto &x : input_shapes_.at(i)) {
          std::cout << x << ", ";
        }
      }
      std::cout << "\nvariables: ";
      for (const auto &x : prep_info_map_.at(group_name).var_names) {
        std::cout << x << ", ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  for (const auto &flav_name : flav_names_) {
    produces<edm::ValueMap<float>>(flav_name);
  }
}

template <typename LeptonType>
void PNETLeptonProducer<LeptonType>::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<LeptonTagInfoCollection> src;
  iEvent.getByToken(src_, src);

  edm::Handle<edm::View<LeptonType>> leps;
  iEvent.getByToken(leps_, leps);

  std::vector<std::vector<float>> mvaScores(flav_names_.size(), std::vector<float>(leps->size(), -1));

  // tagInfo src could be empty if the event has no PV
  if (!src->empty()) {
    assert(src->size() == leps->size());
    for (size_t ilep = 0; ilep < src->size(); ilep++) {
      const auto &taginfo = (*src)[ilep];
      make_inputs(taginfo);
      auto outputs = globalCache()->run(input_names_, data_, input_shapes_)[0];
      // std::cout<<"outputs.size(): "<<outputs.size()<<std::endl;
      // std::cout<<"flav_names_.size(): "<<flav_names_.size()<<std::endl;
      assert(outputs.size() == flav_names_.size());
      for (unsigned int iflav = 0; iflav < flav_names_.size(); ++iflav) {
        mvaScores[iflav][ilep] = outputs.at(iflav);
      }
    }
  }

  for (unsigned int iflav = 0; iflav < flav_names_.size(); ++iflav) {
    std::unique_ptr<edm::ValueMap<float>> pnScore(new edm::ValueMap<float>());
    edm::ValueMap<float>::Filler filler(*pnScore);
    filler.insert(leps, mvaScores[iflav].begin(), mvaScores[iflav].end());
    filler.fill();
    iEvent.put(std::move(pnScore), flav_names_[iflav]);
  }
}

template <typename LeptonType>
std::unique_ptr<cms::Ort::ONNXRuntime> PNETLeptonProducer<LeptonType>::initializeGlobalCache(
    const edm::ParameterSet &cfg) {
  return std::make_unique<cms::Ort::ONNXRuntime>(cfg.getParameter<edm::FileInPath>("model_path").fullPath());
}

template <typename LeptonType>
void PNETLeptonProducer<LeptonType>::globalEndJob(const cms::Ort::ONNXRuntime *cache) {}

template <typename LeptonType>
void PNETLeptonProducer<LeptonType>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input variables");
  desc.add<edm::InputTag>("srcLeps")->setComment("input physics object collection. src and srcLeps must be in synch");
  desc.add<std::vector<std::string>>("flav_names")->setComment("Names of the oputput classes");
  desc.add<std::string>("preprocess_json", "PhysicsTools/NanoAOD/data/PNetMuonId/preprocess.json");
  desc.add<edm::FileInPath>("model_path", edm::FileInPath("PhysicsTools/NanoAOD/data/PNetMuonId/model.onnx"));
  desc.addOptionalUntracked<bool>("debugMode", false);

  std::string modname;
  if (typeid(LeptonType) == typeid(pat::Muon))
    modname += "muon";
  else if (typeid(LeptonType) == typeid(pat::Electron))
    modname += "electron";
  modname += "PNetTags";
  descriptions.add(modname, desc);
}

template <typename LeptonType>
void PNETLeptonProducer<LeptonType>::make_inputs(const DeepBoostedJetFeatures &taginfo) {
  for (unsigned igroup = 0; igroup < input_names_.size(); ++igroup) {
    const auto &group_name = input_names_[igroup];
    const auto &prep_params = prep_info_map_.at(group_name);
    auto &group_values = data_[igroup];
    group_values.resize(input_sizes_[igroup]);

    // first reset group_values to 0
    std::fill(group_values.begin(), group_values.end(), 0);
    unsigned curr_pos = 0;

    // transform/pad
    for (unsigned i = 0; i < prep_params.var_names.size(); ++i) {
      const auto &varname = prep_params.var_names[i];
      const auto &raw_value = taginfo.get(varname);
      const auto &info = prep_params.info(varname);
      int insize = btagbtvdeep::center_norm_pad(raw_value,
                                                info.center,
                                                info.norm_factor,
                                                prep_params.min_length,
                                                prep_params.max_length,
                                                group_values,
                                                curr_pos,
                                                info.pad,
                                                info.replace_inf_value,
                                                info.lower_bound,
                                                info.upper_bound);
      curr_pos += insize;
      if (i == 0 && (!input_shapes_.empty())) {
        input_shapes_[igroup][2] = insize;
      }
    }
    group_values.resize(curr_pos);
  }
}

typedef PNETLeptonProducer<pat::Muon> MuonPNETProducer;
typedef PNETLeptonProducer<pat::Electron> ElectronPNETProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(MuonPNETProducer);
DEFINE_FWK_MODULE(ElectronPNETProducer);
