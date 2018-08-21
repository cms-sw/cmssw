// -*- C++ -*-
//
// Package:    ​RecoBTag/​SecondaryVertex
// Class:      DeepFlavourJetTagsProducer
//
/**\class DeepFlavourJetTagsProducer DeepFlavourJetTagsProducer.cc ​RecoBTag/DeepFlavour/plugins/DeepFlavourJetTagsProducer.cc
 *
 * Description: EDProducer that produces collection of ShallowTagInfos
 *
 * Implementation:
 *    A collection of CandIPTagInfo and CandSecondaryVertexTagInfo and a CombinedSVComputer ESHandle is taken as input and a collection of ShallowTagInfos
 *    is produced as output.
 */
//
// Original Author:  Mauro Verzetti (U. Rochester)
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

//from lwtnn
#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/parse_json.hh"

#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <iostream>

#include <boost/algorithm/string.hpp>
using namespace std;
using namespace reco;
//
// class declaration
//

namespace {

struct MVAVar {
	std::string name;
	reco::btau::TaggingVariableName id;
	int index;
	double default_value;
};

class NeuralNetworkAndConstants {
public:

	NeuralNetworkAndConstants(const edm::ParameterSet&);

	std::unique_ptr<const lwt::LightweightNeuralNetwork> const& neural_network() const { return neural_network_; }
	vector<string> const& outputs() const { return outputs_; }
	bool check_sv_for_defaults() const { return check_sv_for_defaults_; }
	map<string, string> const& toadd() const { return toadd_; }
	vector<MVAVar> const& variables() const { return variables_; }

private:

	std::unique_ptr<const lwt::LightweightNeuralNetwork> neural_network_;
	vector<string> outputs_;
	bool check_sv_for_defaults_;
	map<string, string> toadd_;
	vector<MVAVar> variables_;
};

class DeepFlavourJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<NeuralNetworkAndConstants>> {
public:
	explicit DeepFlavourJetTagsProducer(const edm::ParameterSet&, NeuralNetworkAndConstants const*);
	~DeepFlavourJetTagsProducer() override;

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

	static std::unique_ptr<NeuralNetworkAndConstants> initializeGlobalCache(const edm::ParameterSet& iConfig) {
		return std::make_unique<NeuralNetworkAndConstants>(iConfig);
	}

	static void globalEndJob(NeuralNetworkAndConstants*) { }

private:
	typedef std::vector<reco::ShallowTagInfo> INFOS;
	void beginStream(edm::StreamID) override {}
	void produce(edm::Event&, const edm::EventSetup&) override;
	void endStream() override {}

	// ----------member data ---------------------------
	const edm::EDGetTokenT< INFOS > src_;
	lwt::ValueMap inputs_; //typedef of unordered_map<string, float>
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//

NeuralNetworkAndConstants::NeuralNetworkAndConstants(const edm::ParameterSet& iConfig) :
	check_sv_for_defaults_(iConfig.getParameter<bool>("checkSVForDefaults"))
{
	bool mean_padding = iConfig.getParameter<bool>("meanPadding");

	//parse json
	edm::FileInPath nnconfig = iConfig.getParameter<edm::FileInPath>("NNConfig");
	ifstream jsonfile(nnconfig.fullPath());
	auto config = lwt::parse_json(jsonfile);

	//create NN and store the output names for the future
	neural_network_ = std::make_unique<const lwt::LightweightNeuralNetwork>(config.inputs, config.layers, config.outputs);

	outputs_ = config.outputs;
	set<string> outset(outputs_.begin(), outputs_.end());

	//in case we want to merge some different outputs together
	edm::ParameterSet toaddPSet = iConfig.getParameter<edm::ParameterSet>("toAdd");
	for(auto const& output : toaddPSet.getParameterNamesForType<string>()) {
		string target = toaddPSet.getParameter<string>(output);
		if(outset.find(output) == outset.end())
			throw cms::Exception("RuntimeError") << "The required output: " << output << " to be added to " << target << " could not be found among the NN outputs" << endl;
		if(outset.find(target) == outset.end())
			throw cms::Exception("RuntimeError") << "The required output: " << target << ", target of addition of " << output << " could not be found among the NN outputs" << endl;
		toadd_[output] = target;
	}

	//get the set-up for the inputs
	for(auto const& input : config.inputs) {
		MVAVar var;
		var.name = input.name;
		//two paradigms 
		vector<string> tokens;
		if (var.name != "Jet_JP" && var.name != "Jet_JBP" && var.name != "Jet_SoftMu" && var.name != "Jet_SoftEl"){boost::split(tokens,var.name,boost::is_any_of("_"));}
		else {tokens.push_back(var.name);}
		if(tokens.empty()) {
			throw cms::Exception("RuntimeError") << "I could not parse properly " << input.name << " as input feature" << std::endl;
		}
		var.id = reco::getTaggingVariableName(tokens.at(0));
		//die grafully if the tagging variable is not found!
		if(var.id == reco::btau::lastTaggingVariable) {
			throw cms::Exception("ValueError") << "I could not find the TaggingVariable named " << tokens.at(0) 
				<< " from the NN input variable: " << input.name
				<< ". Please check the spelling" <<  std::endl;
		}
		var.index = (tokens.size() == 2) ? stoi(tokens.at(1)) : -1;
		var.default_value = (mean_padding) ? 0. : -1*input.offset; //set default to -offset so that when scaling (val+offset)*scale the outcome is 0
		//for mean padding it is set to zero so that undefined values are assigned -mean/scale
		
		variables_.push_back(var);
	}
}

DeepFlavourJetTagsProducer::DeepFlavourJetTagsProducer(const edm::ParameterSet& iConfig, NeuralNetworkAndConstants const* gc) :
	src_( consumes< INFOS >(iConfig.getParameter<edm::InputTag>("src")) ),
	inputs_()
{
	//produce one output kind per node
	for(auto const& outnode : gc->outputs())	{
		if(gc->toadd().find(outnode) == gc->toadd().end()){ //produce output only if does not get added
			produces<JetTagCollection>(outnode);
		}
	}
}

DeepFlavourJetTagsProducer::~DeepFlavourJetTagsProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DeepFlavourJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	NeuralNetworkAndConstants const* gc = globalCache();
	vector<string> const& outputs = gc->outputs();
	map<string, string> const& toadd = gc->toadd();

	// get input TagInfos
	edm::Handle<INFOS> taginfos;
	iEvent.getByToken(src_, taginfos);

	// create the output collection
	// which is a "map" RefToBase<Jet> --> float
	vector< std::unique_ptr<JetTagCollection> > output_tags;
	output_tags.reserve(outputs.size());
	for(size_t i = 0; i < outputs.size(); ++i) {
		if(!taginfos->empty()) {
			edm::RefToBase<Jet> jj = taginfos->begin()->jet();
			output_tags.push_back(
				std::make_unique<JetTagCollection>(
					edm::makeRefToBaseProdFrom(jj, iEvent)
					)
				);
		} else {
			output_tags.push_back(
				std::make_unique<JetTagCollection>()
				);			
		}
	}

	int naninput = 0;
	int nanoutput = 0;

	// loop over TagInfos
	for(auto& info : *(taginfos)) {
		//convert the taginfo into the value map in the appropriate way
		TaggingVariableList vars = info.taggingVariables();
    //if there are no tracks there's no point in doing it
		bool notracks = (vars.get(reco::btau::jetNSelectedTracks) == 0); 
		bool novtx = (vars.get(reco::btau::jetNSecondaryVertices) == 0); 
		bool defaulted = (gc->check_sv_for_defaults()) ? (notracks && novtx) : notracks;
		lwt::ValueMap nnout; //returned value

		if(!defaulted) {
			for(auto const& var : gc->variables()) {
				if(var.index >= 0){
					std::vector<float> vals = vars.getList(var.id, false);
					inputs_[var.name] = (((int) vals.size()) > var.index) ? vals.at(var.index) : var.default_value;
				}
				//single value tagging var
				else {
					inputs_[var.name] = vars.get(var.id, var.default_value);
				}

				//count if the input is nan
				if(std::isnan(inputs_[var.name])) {
					naninput++;
				}
			}

			//compute NN output(s)
			nnout = gc->neural_network()->compute(inputs_);
			
			//merge outputs
			for(auto const& entry : toadd) {
				nnout[entry.second] += nnout[entry.first];
			}

			//count if the output is nan
			for(const auto& entry : nnout) {
				if(std::isnan(entry.second)) {
					nanoutput++;
				}
			}
		}

		//ket the maps key
		edm::RefToBase<Jet> key = info.jet();
		
		//dump the NN output(s)
		for(size_t i = 0; i < outputs.size(); ++i) {
			(*output_tags[i])[key] = (defaulted) ? -1 : nnout[outputs[i]];
		}
	}

	if( naninput + nanoutput > 0 ) {
		edm::LogWarning("ValueError") << "The NN encountered " << naninput << " nan input TagInfo values and produced " << nanoutput << " nan output values";
	}

	// put the output in the event
	for(size_t i = 0; i < outputs.size(); ++i) {
		if(toadd.find(outputs[i]) == toadd.end()) {
			iEvent.put(std::move(output_tags[i]), outputs[i]);
		}
	}
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DeepFlavourJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
} // end unnamed namespace

//define this as a plug-in
DEFINE_FWK_MODULE(DeepFlavourJetTagsProducer);
