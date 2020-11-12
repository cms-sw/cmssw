#include "RecoBTag/Combined/interface/HeavyIonCSVTagger.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include <memory>
#include <algorithm>
#include <map>
#include <vector>

HeavyIonCSVTagger::Tokens::Tokens(const edm::ParameterSet &configuration, edm::ESConsumesCollector &&cc) {
  if (configuration.getParameter<bool>("useCondDB")) {
    gbrForest_ = cc.consumes(edm::ESInputTag{"",
                                             configuration.existsAs<std::string>("gbrForestLabel")
                                                 ? configuration.getParameter<std::string>("gbrForestLabel")
                                                 : ""});
  }
}
HeavyIonCSVTagger::HeavyIonCSVTagger(const edm::ParameterSet &configuration, Tokens tokens)
    : sv_computer_(configuration.getParameter<edm::ParameterSet>("sv_cfg")),
      mva_name_(configuration.getParameter<std::string>("mvaName")),
      weight_file_(configuration.getParameter<edm::FileInPath>("weightFile")),
      use_GBRForest_(configuration.getParameter<bool>("useGBRForest")),
      use_adaBoost_(configuration.getParameter<bool>("useAdaBoost")),
      tokens_{tokens} {
  vpset vars_definition = configuration.getParameter<vpset>("variables");

  for (auto &var : vars_definition) {
    MVAVar mva_var;
    mva_var.name = var.getParameter<std::string>("name");
    mva_var.id = reco::getTaggingVariableName(var.getParameter<std::string>("taggingVarName"));

    mva_var.has_index = var.existsAs<int>("idx");
    mva_var.index = mva_var.has_index ? var.getParameter<int>("idx") : 0;
    mva_var.default_value = var.getParameter<double>("default");

    variables_.push_back(mva_var);
  }

  uses(0, "impactParameterTagInfos");
  uses(1, "secondaryVertexTagInfos");
}

void HeavyIonCSVTagger::initialize(const JetTagComputerRecord &record) {
  mvaID_ = std::make_unique<TMVAEvaluator>();

  std::vector<std::string> variable_names;
  variable_names.reserve(variables_.size());

  for (auto &var : variables_) {
    variable_names.push_back(var.name);
  }
  std::vector<std::string> spectators;

  if (tokens_.gbrForest_.isInitialized()) {
    mvaID_->initializeGBRForest(&record.get(tokens_.gbrForest_), variable_names, spectators, use_adaBoost_);
  } else {
    mvaID_->initialize("Color:Silent:Error",
                       mva_name_,
                       weight_file_.fullPath(),
                       variable_names,
                       spectators,
                       use_GBRForest_,
                       use_adaBoost_);
  }
}

HeavyIonCSVTagger::~HeavyIonCSVTagger() {}

/// b-tag a jet based on track-to-jet parameters in the extened info collection
float HeavyIonCSVTagger::discriminator(const TagInfoHelper &tagInfo) const {
  // default value, used if there are no leptons associated to this jet
  const reco::TrackIPTagInfo &ip_info = tagInfo.get<reco::TrackIPTagInfo>(0);
  const reco::SecondaryVertexTagInfo &sv_info = tagInfo.get<reco::SecondaryVertexTagInfo>(1);
  reco::TaggingVariableList vars = sv_computer_(ip_info, sv_info);

  // Loop over input variables
  std::map<std::string, float> inputs;

  bool notTaggable = false;
  std::vector<float> tagValList = vars.getList(reco::btau::trackSip3dSig, false);
  bool noTrack = (tagValList.empty());
  bool noVertex = (vars.get(reco::btau::vertexCategory, -1.0) == 2);

  for (auto &mva_var : variables_) {
    //vectorial tagging variable
    if (mva_var.has_index) {
      std::vector<float> vals = vars.getList(mva_var.id, false);
      inputs[mva_var.name] = (vals.size() > mva_var.index) ? vals[mva_var.index] : mva_var.default_value;
    }
    //single value tagging var
    else {
      inputs[mva_var.name] = vars.get(mva_var.id, mva_var.default_value);
    }
  }

  if (noTrack && noVertex)
    notTaggable = true;

  //get the MVA output
  float tag = (mvaID_->evaluate(inputs) + 1.) / 2.;

  if (notTaggable)
    tag = -1;

  return tag;
}
