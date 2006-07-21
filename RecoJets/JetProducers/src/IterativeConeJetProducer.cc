#include <memory>

#include "RecoJets/JetProducers/interface/IterativeConeJetProducer.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Handle.h"

using namespace std;
using namespace reco;

namespace {
  const bool debug = false;

  bool makeCaloJet (const string& fTag) {
    return fTag == "CaloJet";
  }
  bool makeGenJet (const string& fTag) {
    return fTag == "GenJet";
  }
  bool makeBasicJet (const string& fTag) {
    return fTag == "BasicJet";
  }
}

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.

  IterativeConeJetProducer::IterativeConeJetProducer(edm::ParameterSet const& conf):
    alg_(conf.getParameter<double>("seedThreshold"),
	 conf.getParameter<double>("coneRadius"),
	 conf.getParameter<double>("towerThreshold")),
    src_(conf.getParameter<string>( "src" )),
    jetType_ (conf.getUntrackedParameter<string>( "jetType", "CaloJet"))
  {
    // branch alias
    char label [32];
    sprintf (label, "IC%d%s", 
	     int (floor (conf.getParameter<double>("coneRadius") * 10. + 0.5)), 
	     jetType_.c_str());
    if (makeCaloJet (jetType_)) produces<CaloJetCollection>().setBranchAlias (label);
    if (makeGenJet (jetType_)) produces<GenJetCollection>().setBranchAlias (label);
    if (makeBasicJet (jetType_)) produces<BasicJetCollection>().setBranchAlias (label);
  }

  // Virtual destructor needed.
  IterativeConeJetProducer::~IterativeConeJetProducer() { }  

  // Functions that gets called by framework every event
  void IterativeConeJetProducer::produce(edm::Event& e, const edm::EventSetup&)
  {
    if (debug) {
      std::cout << "IterativeConeJetProducer::produce->" 
	//	      << " run:" <<  e.getRun()
		<< " event:" << e.id ()
		<< " timestamp:" << e.time ().value ()
	//	      << " section:" << e.getLuminositySection()
		<< "\n ====================================================="
		<< std::endl;
    } 
    // get input
    edm::Handle<CandidateCollection> inputs;
    e.getByLabel( src_, inputs );                    
    vector <const Candidate*> input;
    vector <ProtoJet> output;
    // fill input
    input.reserve (inputs->size ());
    CandidateCollection::const_iterator input_object = inputs->begin ();
    for (; input_object != inputs->end (); input_object++) {
      if (debug) {
	std::cout << "IterativeConeJetProducer::produce-> input E/p/eta/phi/M: " << input_object->energy ()
		  << '/' << input_object->p() << '/' << input_object->eta ()
		  << '/' << input_object->phi() << '/' << input_object->mass ()
		  << std::endl;
      }
      input.push_back (&*input_object); 
    }
    // run algorithm
    alg_.run (input, &output);
    // produce output collection
    auto_ptr<CaloJetCollection> caloJets;
    if (makeCaloJet (jetType_)) caloJets.reset (new CaloJetCollection);
    auto_ptr<GenJetCollection> genJets;
    if (makeGenJet (jetType_)) genJets.reset (new GenJetCollection);
    auto_ptr<BasicJetCollection> basicJets;
    if (makeBasicJet (jetType_)) basicJets.reset (new BasicJetCollection);

    vector <ProtoJet>::const_iterator protojet = output.begin ();
    JetMaker jetMaker;
    for (; protojet != output.end (); protojet++) {
      if (caloJets.get ()) {
	caloJets->push_back (jetMaker.makeCaloJet (*protojet));
	if (debug) std::cout << "IterativeConeJetProducer::produce-> add protojet to CaloJets." << std::endl;
      }
      if (genJets.get ()) { 
	genJets->push_back (jetMaker.makeGenJet (*protojet));
	if (debug) std::cout << "IterativeConeJetProducer::produce-> add protojet to GenJets." << std::endl;
      }
      if (basicJets.get ()) { 
	basicJets->push_back (jetMaker.makeBasicJet (*protojet));
	if (debug) std::cout << "IterativeConeJetProducer::produce-> add protojet to BasicJets." << std::endl;
      }
    }
    // store output
    if (caloJets.get ()) e.put(caloJets);  //Puts Jet Collection into event
    if (genJets.get ()) e.put(genJets);  //Puts Jet Collection into event
    if (basicJets.get ()) e.put(basicJets);  //Puts Jet Collection into event
  }

}
