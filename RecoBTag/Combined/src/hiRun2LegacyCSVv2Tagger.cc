#include "RecoBTag/Combined/interface/hiRun2LegacyCSVv2Tagger.h"

#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"

#include <iostream>
#include <memory>

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

const bool PbPbdebug = false;

hiRun2LegacyCSVv2Tagger::Tokens::Tokens(const edm::ParameterSet &configuration, edm::ESConsumesCollector &&cc) {
  if (configuration.getParameter<bool>("useCondDB")) {
    gbrForest_ = cc.consumes(edm::ESInputTag{"",
                                             configuration.existsAs<std::string>("gbrForestLabel")
                                                 ? configuration.getParameter<std::string>("gbrForestLabel")
                                                 : ""});
  }
}

hiRun2LegacyCSVv2Tagger::hiRun2LegacyCSVv2Tagger(const edm::ParameterSet &configuration, Tokens tokens)
    : sl_computer_(configuration.getParameter<edm::ParameterSet>("slComputerCfg")),
      sv_computer_(configuration.getParameter<edm::ParameterSet>("slComputerCfg")),
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

    if (PbPbdebug)
      std::cout << "hiRun2LegacyCSVv2:mva_var.id =" << mva_var.id
                << " var.getParameter(taggingVarName)=" << var.getParameter<std::string>("taggingVarName")
                << " index = " << mva_var.index << std::endl;

    variables_.push_back(mva_var);
  }

  uses(0, "impactParameterTagInfos");
  uses(1, "secondaryVertexTagInfos");
}

void hiRun2LegacyCSVv2Tagger::initialize(const JetTagComputerRecord &record) {
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

hiRun2LegacyCSVv2Tagger::~hiRun2LegacyCSVv2Tagger() {}

/// b-tag a jet based on track-to-jet parameters in the extened info collection
float hiRun2LegacyCSVv2Tagger::discriminator(const TagInfoHelper &tagInfo) const {
  // default value, used if there are no leptons associated to this jet
  const reco::TrackIPTagInfo &ip_info = tagInfo.get<reco::TrackIPTagInfo>(0);
  const reco::SecondaryVertexTagInfo &sv_info = tagInfo.get<reco::SecondaryVertexTagInfo>(1);
  reco::TaggingVariableList vars = sv_computer_(ip_info, sv_info);

  // Loop over input variables
  std::map<std::string, float> inputs;

  //For debugging;
  float save_pt_value = -1.0;
  float save_eta_value = -999.0;
  bool passes_cuts = false;

  bool notTaggable = false;
  bool noTrack = false;
  bool printdebug = false;
  float vtxMassVal = 0.;

  for (auto &mva_var : variables_) {
    //vectorial tagging variable
    if (mva_var.has_index) {
      std::vector<float> vals = vars.getList(mva_var.id, false);
      inputs[mva_var.name] = (vals.size() > mva_var.index) ? vals[mva_var.index] : mva_var.default_value;

      if (mva_var.name == "TagVarCSV_trackSip3dSig_0" && inputs[mva_var.name] < -98.999)
        noTrack = true;
      if (passes_cuts) {
        if (printdebug)
          std::cout << inputs[mva_var.name] << "\t";
      }

      if (mva_var.name == "Jet_pt") {
        save_pt_value = inputs[mva_var.name];
      }

      if (mva_var.name == "Jet_eta") {
        save_eta_value = inputs[mva_var.name];
        passes_cuts = (save_pt_value > 30 && save_eta_value > -2.4 && save_eta_value < 2.4);
        if (printdebug) {
          if (passes_cuts)
            std::cout << save_pt_value << "\t" << save_eta_value << "\t";
        }
      }

    }
    //single value tagging var
    else {
      inputs[mva_var.name] = vars.get(mva_var.id, mva_var.default_value);

      //IK: vtxMass check to check vtxType: vtxType = 2 (no vtx), vtxMass < 0, vtxType = 1 (pseudo vtx), vtxMass > 0
      if (mva_var.name == "TagVarCSV_vertexMass") {
        vtxMassVal = inputs[mva_var.name];
      }

      if (passes_cuts) {
        if (printdebug)
          std::cout << inputs[mva_var.name] << "\t";
      }
    }
  }

  //IK: if no reco vtx (including pseudo vtx) and no tracks passing all selections (including K0s veto) -> jet is not taggable

  if (vtxMassVal < 0 && noTrack) {
    notTaggable = true;
  }

  //get the MVA output
  float tag = (mvaID_->evaluate(inputs) + 1) / 2.;
  if (printdebug) {
    if (passes_cuts)
      std::cout << tag << "\n";
  }

  if (notTaggable) {
    tag = -1;
    if (PbPbdebug)
      std::cout << " --- jet not taggable" << std::endl;
  }

  if (PbPbdebug) {
    std::cout << "Looking at a jet of " << save_pt_value << " GeV" << std::endl;
    for (const auto &x : inputs)
      std::cout << "Variable = " << x.first << " value = " << x.second << std::endl;
    std::cout << "  ---  Result : " << tag << std::endl;
  }

  return tag;
}
