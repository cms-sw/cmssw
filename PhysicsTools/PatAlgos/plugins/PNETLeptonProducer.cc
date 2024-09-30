// -*- C++ -*-
//
// Package:    PhysicsTools/PatAlgos
// Class:      PNETLeptonProducer
//
/**\class PNETLeptonProducer PNETLeptonProducer.cc PhysicsTools/PatAlgos/plugins/PNETLeptonProducer.cc


*/
//
// Original Author:  Sergio Sanchez Cruz
//         Created:  Mon, 15 May 2023 08:32:03 GMT
//
//

#include "PhysicsTools/PatAlgos/interface/PNETLeptonProducer.h"
#include "PhysicsTools/PatAlgos/interface/LeptonTagInfoCollectionProducer.h"

template <typename T>
PNETLeptonProducer<T>::PNETLeptonProducer(const edm::ParameterSet &iConfig, const cms::Ort::ONNXRuntime *cache)
    : src_(consumes<pat::LeptonTagInfoCollection<T>>(iConfig.getParameter<edm::InputTag>("src"))),
      leps_(consumes<std::vector<T>>(iConfig.getParameter<edm::InputTag>("srcLeps"))),
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
    produces<edm::ValueMap<float>>("pnScore" + flav_name);
  }
}

template <typename T>
void PNETLeptonProducer<T>::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<pat::LeptonTagInfoCollection<T>> src;
  iEvent.getByToken(src_, src);

  edm::Handle<std::vector<T>> leps;
  iEvent.getByToken(leps_, leps);

  std::vector<std::vector<float>> mvaScores;
  for (unsigned int iflav = 0; iflav < flav_names_.size(); ++iflav) {
    mvaScores.push_back(std::vector<float>());
  }

  for (size_t ilep = 0; ilep < src->size(); ilep++) {
    const pat::LeptonTagInfo<T> taginfo = (*src)[ilep];
    make_inputs(taginfo);
    std::vector<float> outputs;
    outputs = globalCache()->run(input_names_, data_, input_shapes_)[0];
    // std::cout<<"outputs.size(): "<<outputs.size()<<std::endl;
    // std::cout<<"flav_names_.size(): "<<flav_names_.size()<<std::endl;
    assert(outputs.size() == flav_names_.size());
    for (unsigned int iflav = 0; iflav < flav_names_.size(); ++iflav) {
      mvaScores[iflav].push_back(outputs.at(iflav));
    }
  }

  for (unsigned int iflav = 0; iflav < flav_names_.size(); ++iflav) {
    std::unique_ptr<edm::ValueMap<float>> pnScore(new edm::ValueMap<float>());
    edm::ValueMap<float>::Filler filler(*pnScore);
    filler.insert(leps, mvaScores[iflav].begin(), mvaScores[iflav].end());
    filler.fill();
    iEvent.put(std::move(pnScore), "pnScore" + flav_names_[iflav]);
  }
}

template <typename T>
std::unique_ptr<cms::Ort::ONNXRuntime> PNETLeptonProducer<T>::initializeGlobalCache(const edm::ParameterSet &cfg) {
  return std::make_unique<cms::Ort::ONNXRuntime>(cfg.getParameter<edm::FileInPath>("model_path").fullPath());
}

template <typename T>
void PNETLeptonProducer<T>::globalEndJob(const cms::Ort::ONNXRuntime *cache) {}

template <typename T>
edm::ParameterSetDescription PNETLeptonProducer<T>::getDescription() {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input variables");
  desc.add<edm::InputTag>("srcLeps")->setComment("input physics object collection. src and srcLeps must be in synch");
  desc.add<std::vector<std::string>>("flav_names")->setComment("Names of the oputput classes");
  desc.add<std::string>("preprocess_json", "");
  desc.add<std::string>("name")->setComment("output score variable name");
  desc.add<edm::FileInPath>("model_path", edm::FileInPath("PhysicsTools/PatAlgos/data/model.onnx"));
  desc.addOptionalUntracked<bool>("debugMode", false);

  return desc;
}

template <typename T>
void PNETLeptonProducer<T>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc = getDescription();
  std::string modname;
  if (typeid(T) == typeid(pat::Jet))
    modname += "Jet";
  else if (typeid(T) == typeid(pat::Muon))
    modname += "Muon";
  else if (typeid(T) == typeid(pat::Electron))
    modname += "Ele";
  modname += "PNETLeptonProducer";
  descriptions.add(modname, desc);
}

template <typename T>
void PNETLeptonProducer<T>::make_inputs(const pat::LeptonTagInfo<T> &taginfo) {
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
      const auto &raw_value = taginfo.features().get(varname);
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

typedef PNETLeptonProducer<pat::Muon> PNETMuonProducer;
typedef PNETLeptonProducer<pat::Electron> PNETElectronProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(PNETMuonProducer);
DEFINE_FWK_MODULE(PNETElectronProducer);
