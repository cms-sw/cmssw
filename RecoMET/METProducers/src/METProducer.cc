// File: METProducer.cc 
// Description:  see METProducer.h
// Author: R. Cavanaugh, The University of Florida
// Creation Date:  20.04.2006.
//
//--------------------------------------------
#include <memory>
#include "RecoMET/METProducers/interface/METProducer.h"
#include "RecoMET/METAlgorithms/interface/SignCaloSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "RecoMET/METAlgorithms/interface/CaloSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/GenSpecificAlgo.h"
//#include "DataFormats/METObjects/interface/METCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats//Common/interface/View.h"

using namespace edm;
using namespace std;
using namespace reco;

namespace cms 
{
  //--------------------------------------------------------------------------
  // Constructor : used to fill the parameters from the configuration file
  // Currently there are only two defined parameters:
  // 1. src = the label of the input data product (which must derive from 
  //    Candidate)
  // 2. METType = the type of to produce into the event.  currently there are
  //    only two types of MET defined: (1) MET from calorimetery (and so 
  //    contains extra information specific to calorimetery) and (2) the 
  //    default MET which contains only generic information.  Additional
  //    MET types will appear (such as GenMET) in the future.  (All "types"
  //    of MET inherit from RecoCandidate and merely extend that class with
  //    extra information)
  //-----------------------------------
  METProducer::METProducer(const edm::ParameterSet& iConfig) : conf_(iConfig),alg_() 
  {
    inputLabel = iConfig.getParameter<edm::InputTag>("src");
    inputType  = iConfig.getParameter<std::string>("InputType");
    METtype    = iConfig.getParameter<std::string>("METType");
    alias      = iConfig.getParameter<std::string>("alias");
    globalThreshold = iConfig.getParameter<double>("globalThreshold");
    noHF = iConfig.getParameter<bool>("noHF");

    if(      METtype == "CaloMET" ) 
      produces<CaloMETCollection>().setBranchAlias(alias.c_str()); 
    else if( METtype == "GenMET" )  
      produces<GenMETCollection>().setBranchAlias(alias.c_str());
    else if (METtype == "CaloMETSignif")
      produces<CaloMETCollection>().setBranchAlias(alias.c_str());
    else                            
      produces<METCollection>().setBranchAlias(alias.c_str()); 
  }
  //--------------------------------------------------------------------------

  //--------------------------------------------------------------------------
  // Default Constructor
  //-----------------------------------
  METProducer::METProducer() : alg_() 
  {
    produces<METCollection>(); 
  }
  //--------------------------------------------------------------------------

  //--------------------------------------------------------------------------
  // Default Destructor
  //-----------------------------------
  METProducer::~METProducer() {}
  //--------------------------------------------------------------------------

  //--------------------------------------------------------------------------
  // Run Algorithm and put results into event
  //-----------------------------------
  void METProducer::produce(Event& event, const EventSetup& setup) 
  {

    //-----------------------------------
    // Step A: Get Inputs.  Create an empty collection of candidates
    edm::Handle<edm::View<Candidate> > input;
    event.getByLabel(inputLabel,input);
    //-----------------------------------
    // Step B: Create an empty MET struct output.
    CommonMETData output;
    /*
    //-----------------------------------
    // Step C: Convert input source to type CandidateCollection
    const RefToBaseVector<Candidate> inputCol = inputHandle->refVector();
    const CandidateCollection *input = (const CandidateCollection *)inputCol.product();
    */
    //-----------------------------------
    // Step C2: Invoke the MET algorithm, which runs on any CandidateCollection input. 
    alg_.run(input, &output, globalThreshold);
    //-----------------------------------
    // Step D: Invoke the specific "afterburner", which adds information
    //         depending on the input type, given via the config parameter.
    //         Also, after the specific algorithm has been called, store
    //         the output into the Event.
    if( METtype == "CaloMET" ) 
    {
      CaloSpecificAlgo calo;
      std::auto_ptr<CaloMETCollection> calometcoll; 
      calometcoll.reset(new CaloMETCollection);
      calometcoll->push_back( calo.addInfo(input, output, noHF) );
      event.put( calometcoll );
    }
    //-----------------------------------
    else if( METtype == "GenMET" ) 
    {
      GenSpecificAlgo gen;
      std::auto_ptr<GenMETCollection> genmetcoll;
      genmetcoll.reset (new GenMETCollection);
      genmetcoll->push_back( gen.addInfo(input, output) );
      event.put( genmetcoll );
    }
    //-----------------------------------
    else if( METtype == "CaloMETSignif" ) 
    {
      SignCaloSpecificAlgo calo;
      // first calculate all standard info. Then over-write the values used by the significance:
      std::auto_ptr<CaloMETCollection> calometcoll; 
      calometcoll.reset(new CaloMETCollection);
      metsig::SignAlgoResolutions resolutions(conf_);
      calometcoll->push_back( calo.addInfo(input, output, resolutions, noHF, globalThreshold) );
      event.put( calometcoll );
    }
    //-----------------------------------
    else
    {
      LorentzVector p4( output.mex, output.mey, 0.0, output.met);
      Point vtx(0,0,0);
      MET met( output.sumet, p4, vtx );
      std::auto_ptr<METCollection> metcoll;
      metcoll.reset(new METCollection);
      metcoll->push_back( met );
      event.put( metcoll );
    }
    //-----------------------------------
  }
  //--------------------------------------------------------------------------
}
