// File: METProducer.cc
// Description:  see METProducer.h
// Author: R. Cavanaugh, The University of Florida
// Creation Date:  20.04.2006.
//
//--------------------------------------------
#include <memory>
#include "RecoMET/METProducers/interface/METProducer.h"
//#include "DataFormats/METObjects/interface/METCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Handle.h"

using namespace edm;
using namespace std;
using namespace reco;

namespace cms 
{

  METProducer::METProducer(const edm::ParameterSet& iConfig) : alg_() 
  {
    produces<METCollection>(); 
    inputLabel = iConfig.getParameter<std::string>("src");
  }

  METProducer::METProducer() : alg_() 
  {
    produces<METCollection>(); 
  }

  METProducer::~METProducer() {}

  void METProducer::produce(Event& event, const EventSetup& setup) 
  {
    // Step A: Get Inputs.
    edm::Handle<CandidateCollection> inputs;
    event.getByLabel( inputLabel, inputs );
    vector <const Candidate*> input;
    input.reserve( inputs->size() );
    CandidateCollection::const_iterator input_object = inputs->begin();
    for( ; input_object != inputs->end(); input_object++ )
      {
	input.push_back( &*input_object );
      }
    // Step B: Create an empty output.
    std::auto_ptr<METCollection> result(new METCollection);
    // Step C: Invoke the algorithm. 
    alg_.run(input, *result);
    // Step D: Put output into the Event.
    event.put(result);
  }
}
