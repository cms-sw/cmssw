#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

#include "RecoBTag/CTagging/interface/CharmTagger.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <iostream>

#include "TDirectory.h" //DEBUG

CharmTagger::CharmTagger(const edm::ParameterSet & configuration):
	sl_computer_(configuration.getParameter<edm::ParameterSet>("slComputerCfg"))
{
	uses(0, "pfImpactParameterTagInfos");
	uses(1, "pfInclusiveSecondaryVertexFinderCtagLTagInfos");
	uses(2, "softPFMuonsTagInfos");
	uses(3, "softPFElectronsTagInfos");

	edm::FileInPath weight_file=configuration.getParameter<edm::FileInPath>("weightFile");
	mvaID_.reset(new TMVAEvaluator());
	
	vpset vars_def = configuration.getParameter<vpset>("variables");
	std::vector<std::string> variable_names;
	variable_names.reserve(vars_def.size());

	for(auto &var : vars_def) {
		variable_names.push_back(
			var.getParameter<std::string>("name")
			);

		MVAVar mva_var;
		mva_var.name = var.getParameter<std::string>("name");
		mva_var.id = reco::getTaggingVariableName(
			var.getParameter<std::string>("taggingVarName")
			);
		mva_var.has_index = var.existsAs<int>("idx") ;
		mva_var.index = mva_var.has_index ? var.getParameter<int>("idx") : 0;
		mva_var.default_value = var.getParameter<double>("default");

		variables_.push_back(mva_var);
	}
	std::vector<std::string> spectators;
	
	mvaID_->initialize(
		"Color:Silent:Error", 
		"BDT", 
		weight_file.fullPath(), 
		variable_names, spectators,
		true, //useGBRForest
		false  //useAdaBoost (output normalized between 0 and 1)
		);

  //DEBUG
	debug_mode_ = configuration.existsAs<std::string>("debugFile");
	if(debug_mode_){
		std::cout << "variable_names has " << variable_names.size() << " names" << std::endl;
		ext_file_.reset(new TFile(configuration.getParameter<std::string>("debugFile").c_str(), "recreate")); //DEBUG
		variable_names.push_back("jetPt" ); 
		variable_names.push_back("jetEta"); 
		variable_names.push_back("vertexCategory"); 
		std::stringstream ntnames;
		bool first=true;
		for(auto &vname : variable_names){
			if(!first) ntnames << ":";
			first = false;
			ntnames << vname;
		}
		TDirectory *context = gDirectory;
		ext_file_->cd();
		tree_ = new TNtuple("tree", "flat support tree", ntnames.str().c_str()); //DEBUG
		context->cd();
	}
}

CharmTagger::~CharmTagger()//DEBUG
{
	if(debug_mode_){
		TDirectory *context = gDirectory;
		ext_file_->cd();
		tree_->Write();
		ext_file_->Close();
		context->cd();
	}
}

/// b-tag a jet based on track-to-jet parameters in the extened info collection
float CharmTagger::discriminator(const TagInfoHelper & tagInfo) const {
  // default value, used if there are no leptons associated to this jet
  const reco::CandIPTagInfo & ip_info = tagInfo.get<reco::CandIPTagInfo>(0);//"pfImpactParameterTagInfos");
	const reco::CandSecondaryVertexTagInfo & sv_info = tagInfo.get<reco::CandSecondaryVertexTagInfo>(1);//"pfInclusiveSecondaryVertexFinderCtagLTagInfos");
	const reco::CandSoftLeptonTagInfo& softmu_info = tagInfo.get<reco::CandSoftLeptonTagInfo>(2);//"softPFMuonsTagInfos");
	const reco::CandSoftLeptonTagInfo& softel_info = tagInfo.get<reco::CandSoftLeptonTagInfo>(3);//"softPFElectronsTagInfos");
	reco::TaggingVariableList vars = sl_computer_(ip_info, sv_info, softmu_info, softel_info);

	// Loop over input variables
	std::map<std::string, float> inputs;
	std::vector<float> debug_values; // DEBUG
	debug_values.reserve(variables_.size()+2);  // DEBUG
	for(auto &mva_var : variables_){
		//vectorial tagging variable
		if(mva_var.has_index){
			std::vector<float> vals = vars.getList(mva_var.id, false);
			inputs[mva_var.name] = (vals.size() > mva_var.index) ? vals[mva_var.index] : mva_var.default_value;
		}
		//single value tagging var
		else {
			inputs[mva_var.name] = vars.get(mva_var.id, mva_var.default_value);
		}
		debug_values.push_back(inputs[mva_var.name]);
	}
	if(debug_mode_){
		debug_values.push_back(vars.get(reco::btau::TaggingVariableName::jetPt) ); //DEBUG
		debug_values.push_back(vars.get(reco::btau::TaggingVariableName::jetEta)); //DEBUG
		debug_values.push_back(vars.get(reco::btau::TaggingVariableName::vertexCategory, 99)); //DEBUG
		tree_->Fill(&debug_values[0]);
	}

  // TMVAEvaluator is not thread safe
 	std::lock_guard<std::mutex> lock(mutex_);
	//get the MVA output
	float tag = mvaID_->evaluate(inputs);
	return tag;
}
