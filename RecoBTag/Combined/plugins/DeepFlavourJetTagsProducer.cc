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
#include <vector>
#include <string>
#include <iostream>

#include <boost/algorithm/string.hpp>
using namespace std;
using namespace reco;
//
// class declaration
//

class DeepFlavourJetTagsProducer : public edm::stream::EDProducer<> {
public:
	explicit DeepFlavourJetTagsProducer(const edm::ParameterSet&);
	~DeepFlavourJetTagsProducer();

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

	struct MVAVar {
		std::string name;
		reco::btau::TaggingVariableName id;
		int index;
		double default_value;
	};

private:
	typedef std::vector<reco::ShallowTagInfo> INFOS;
	virtual void beginStream(edm::StreamID) override {}
	virtual void produce(edm::Event&, const edm::EventSetup&) override;
	virtual void endStream() override {}

	// ----------member data ---------------------------
	const edm::EDGetTokenT< INFOS > src_;
	edm::FileInPath nnconfig_;
	lwt::LightweightNeuralNetwork *neural_network_;
	lwt::ValueMap inputs_; //typedef of unordered_map<string, float>
	vector<string> outputs_;
	vector<MVAVar> variables_;
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
DeepFlavourJetTagsProducer::DeepFlavourJetTagsProducer(const edm::ParameterSet& iConfig) :
  src_( consumes< INFOS >(iConfig.getParameter<edm::InputTag>("src")) ),
	nnconfig_(iConfig.getParameter<edm::FileInPath>("NNConfig")),
	neural_network_(NULL),
	inputs_(),
	outputs_(),
	variables_()
{
	//parse json
	ifstream jsonfile(nnconfig_.fullPath());
	auto config = lwt::parse_json(jsonfile);

	//create NN and store the output names for the future
	neural_network_ = new lwt::LightweightNeuralNetwork(config.inputs, config.layers, config.outputs);
	outputs_ = config.outputs;

	//produce one output kind per node 
	for(auto outnode : config.outputs)	{
		produces<JetTagCollection>(outnode);
	}

	//get the set-up for the inputs
	for(auto& input : config.inputs) {
		MVAVar var;
		var.name = input.name;
		//two paradigms 
		vector<string> tokens;
		if (var.name != "Jet_JP" && var.name != "Jet_JBP" && var.name != "Jet_SoftMu" && var.name != "Jet_SoftEl"){boost::split(tokens,var.name,boost::is_any_of("_"));}
		else {tokens.push_back(var.name);}
		if(!tokens.size()) {
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
		var.default_value = -1*input.offset; //set default to -offset so that when scaling (val+offset)*scale the outcome is 0
		
		variables_.push_back(var);
	}
}


DeepFlavourJetTagsProducer::~DeepFlavourJetTagsProducer()
{
	delete neural_network_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DeepFlavourJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	// get input TagInfos
	edm::Handle<INFOS> taginfos;
	iEvent.getByToken(src_, taginfos);

	// create the output collection
	// which is a "map" RefToBase<Jet> --> float
	vector< std::unique_ptr<JetTagCollection> > output_tags;
	output_tags.reserve(outputs_.size());
	for(size_t i=0; i<outputs_.size(); ++i) {
		if(taginfos->size() > 0) {
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

	// loop over TagInfos
	for(auto& info : *(taginfos)) {
		//convert the taginfo into the value map in the appropriate way
		TaggingVariableList vars = info.taggingVariables();
    //if there are no tracks there's no point in doing it
		bool notracks = (vars.get(reco::btau::jetNSelectedTracks) == 0); 
		lwt::ValueMap nnout; //returned value

		if(!notracks) {
			for(auto& var : variables_) {
				if(var.index >= 0){
					std::vector<float> vals = vars.getList(var.id, false);
					inputs_[var.name] = (((int) vals.size()) > var.index) ? vals.at(var.index) : var.default_value;
				}
				//single value tagging var
				else {
					inputs_[var.name] = vars.get(var.id, var.default_value);
				}
			}

			//compute NN output(s)
			nnout = neural_network_->compute(inputs_);
		}

		//ket the maps key
		edm::RefToBase<Jet> key = info.jet();
		
		//dump the NN output(s)
		for(size_t i=0; i<outputs_.size(); ++i) {
			(*output_tags[i])[key] = (notracks) ? -1 : nnout[outputs_[i]];
			//std::cout << i << ": " << nnout[outputs_[i]] << std::endl;
		}
	}

	// put the output in the event
	for(size_t i=0; i<outputs_.size(); ++i) {
		iEvent.put(std::move(output_tags[i]), outputs_[i]);
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

//define this as a plug-in
DEFINE_FWK_MODULE(DeepFlavourJetTagsProducer);
