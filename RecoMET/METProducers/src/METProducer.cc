// File: METProducer.cc
// Description:  see METProducer.h
// Author: R. Cavanaugh, The University of Florida
// Creation Date:  20.04.2006.
//
//--------------------------------------------
#include <memory>
#include "RecoMET/METProducers/interface/METProducer.h"
#include "RecoMET/METAlgorithms/interface/CaloSpecificAlgo.h"
//#include "DataFormats/METObjects/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
//#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace edm;
using namespace std;
using namespace reco;

namespace cms 
{

  METProducer::METProducer(const edm::ParameterSet& iConfig) : alg_() 
  {
    inputLabel = iConfig.getParameter<std::string>("src");
    METtype    = iConfig.getParameter<std::string>("METType");
    std::cout << "MET Type = " << METtype << std::endl;
    if(      METtype == "CaloMET" ) produces<CaloMETCollection>(); 
    else if( METtype == "GenMET" )  /*produces<GenMETCollection>()*/; 
    else                            produces<METCollection>();
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
      input.push_back( &*input_object );
    // Step B: Create an empty output.
    CommonMETData output;
    // Step C: Invoke the common algorithm, which runs on _any_ candidate input. 
    alg_.run(input, &output);  
    // Step D: Invoke the specific afterburner, which adds information
    //         depending on the input type
    //         Put output into the Event.
    if( METtype == "CaloMET" ) 
    {
      CaloSpecificAlgo calo;
      std::auto_ptr<CaloMETCollection> calometcoll; 
      calometcoll.reset(new CaloMETCollection);
      calometcoll->push_back( calo.addInfo(output) );
      event.put( calometcoll );
    }
    else if( METtype == "GenMET" ) 
    {
    /*
      std::auto_ptr<GenMETCollection> GenMET;
      GenMET.reset (new GenMETCollection);
      GenSpecificInfo gen;
      GenMET->push_back( gen.addInfo(output) );
      event.put(GenMET);
    */
    }
    else
    {
      LorentzVector p4( output.mex, output.mey, output.mez, output.met);
      Point vtx(0,0,0);
      MET met( output.sumet, p4, vtx );
      std::auto_ptr<METCollection> metcoll;
      metcoll.reset(new METCollection);
      metcoll->push_back( met );
      event.put( metcoll );
    }
  }
}
