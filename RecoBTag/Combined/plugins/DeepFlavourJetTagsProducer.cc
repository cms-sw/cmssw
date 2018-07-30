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
	bool check_sv_for_defaults_;
	bool mean_padding_;
	lwt::LightweightNeuralNetwork *neural_network_;
	lwt::ValueMap inputs_; //typedef of unordered_map<string, float>
	vector<string> outputs_;
	vector<MVAVar> variables_;
	map<string, string> toadd_;
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
	check_sv_for_defaults_(iConfig.getParameter<bool>("checkSVForDefaults")),
	mean_padding_(iConfig.getParameter<bool>("meanPadding")),
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
	set<string> outset(outputs_.begin(), outputs_.end());

	//in case we want to merge some different outputs together
	edm::ParameterSet toadd = iConfig.getParameter<edm::ParameterSet>("toAdd");
	for(auto output : toadd.getParameterNamesForType<string>()) {		
		string target = toadd.getParameter<string>(output);
		if(outset.find(output) == outset.end())
			throw cms::Exception("RuntimeError") << "The required output: " << output << " to be added to " << target << " could not be found among the NN outputs" << endl;
		if(outset.find(target) == outset.end())
			throw cms::Exception("RuntimeError") << "The required output: " << target << ", target of addition of " << output << " could not be found among the NN outputs" << endl;
		toadd_[output] = target;
	}

	//produce one output kind per node 	
	for(auto outnode : config.outputs)	{
		if(toadd_.find(outnode) == toadd_.end()){ //produce output only if does not get added
			produces<JetTagCollection>(outnode);
		}
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
		var.default_value = (mean_padding_) ? 0. : -1*input.offset; //set default to -offset so that when scaling (val+offset)*scale the outcome is 0
		//for mean padding it is set to zero so that undefined values are assigned -mean/scale
		
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

	int naninput = 0;
	int nanoutput = 0; 
	// loop over TagInfos
	for(auto& info : *(taginfos)) {
		//convert the taginfo into the value map in the appropriate way
		TaggingVariableList vars = info.taggingVariables();
    //if there are no tracks there's no point in doing it
		bool notracks = (vars.get(reco::btau::jetNSelectedTracks) == 0); 
		bool novtx = (vars.get(reco::btau::jetNSecondaryVertices) == 0); 
		bool defaulted = (check_sv_for_defaults_) ? (notracks && novtx) : notracks;
		lwt::ValueMap nnout; //returned value

		if(!defaulted) {
			for(auto& var : variables_) {
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
			nnout = neural_network_->compute(inputs_);
			
			//merge outputs
			for(auto entry : toadd_) {
				nnout[entry.second] += nnout[entry.first];
			}

			//count if the output is nan
			for(const auto& entry: nnout) {
				if(std::isnan(entry.second)) {
					nanoutput++;
				}
			}
		}

		//ket the maps key
		edm::RefToBase<Jet> key = info.jet();
		
		//dump the NN output(s)
		for(size_t i=0; i<outputs_.size(); ++i) {
			(*output_tags[i])[key] = (defaulted) ? -1 : nnout[outputs_[i]];
		}
	}

	if( naninput + nanoutput > 0 ) {
		edm::LogWarning("ValueError") << "The NN encountered " << naninput << " nan input TagInfo values and produced " << nanoutput << " nan output values";
	}

	// put the output in the event
	for(size_t i=0; i<outputs_.size(); ++i) {
		if(toadd_.find(outputs_[i]) == toadd_.end()) {
			iEvent.put(std::move(output_tags[i]), outputs_[i]);
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

//define this as a plug-in
DEFINE_FWK_MODULE(DeepFlavourJetTagsProducer);
