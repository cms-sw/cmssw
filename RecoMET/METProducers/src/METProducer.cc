// File: METProducer.cc 
// Description:  see METProducer.h
// Author: R. Cavanaugh, The University of Florida
// Creation Date:  20.04.2006.
//
//--------------------------------------------
// Modification by R. Remington on 10/21/08
// Added globalThreshold input Parameter to impose on each tower in tower collection
// that is looped over by the CaloSpecificAlgo.  This is in order to fulfill Scheme B threhsolds...   
// Modified:     12.13.2008 by R.Cavanaugh, UIC/Fermilab
// Description:  include Particle Flow MET
// Modified:     12.12.2008  by R. Remington, UFL
// Description:  include TCMET , move alg_.run() inside of relevant if-statements, and add METSignficance algorithm to METtype="CaloMET" cases

#include <memory>
#include "RecoMET/METProducers/interface/METProducer.h"
#include "RecoMET/METAlgorithms/interface/SignCaloSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "RecoMET/METAlgorithms/interface/CaloSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/GenSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/TCMETAlgo.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/View.h"

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
    else if( METtype == "PFMET" )
      produces<PFMETCollection>().setBranchAlias(alias.c_str()); 
    else if (METtype == "TCMET" )
      {
	produces<METCollection>().setBranchAlias(alias.c_str());
	TCMETAlgo ALGO;
	responseFunction_ = (*ALGO.getResponseFunction());
      }
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

    //    alg_.run(input, &output, globalThreshold);   // No need to run this for all METTypes!
 
    //-----------------------------------
    // Step D: Invoke the specific "afterburner", which adds information
    //         depending on the input type, given via the config parameter.
    //         Also, after the specific algorithm has been called, store
    //         the output into the Event.

    if( METtype == "CaloMET" ) 
    {
      /*
	//Old implementation (prior to 30X and 11/14/2008)  
      alg_.run(input, &output, globalThreshold); 
      CaloSpecificAlgo calo;
      std::auto_ptr<CaloMETCollection> calometcoll; 
      calometcoll.reset(new CaloMETCollection);
      calometcoll->push_back( calo.addInfo(input, output, noHF, globalThreshold) );
      event.put( calometcoll );
      */
      
      //Run Basic MET Algorithm
      alg_.run(input, &output, globalThreshold);

      // Run CaloSpecific Algorithm
      CaloSpecificAlgo calospecalgo;
      CaloMET calomet = calospecalgo.addInfo(input,output,noHF, globalThreshold);

      //Run algorithm to calculate CaloMET Significance and add to the CaloMET Object
      SignCaloSpecificAlgo signcalospecalgo;
      metsig::SignAlgoResolutions resolutions(conf_);
      calomet.SetMetSignificance( signcalospecalgo.addSignificance(input,output,resolutions,noHF,globalThreshold) );

      //Store CaloMET object in CaloMET collection 
      std::auto_ptr<CaloMETCollection> calometcoll;
      calometcoll.reset(new CaloMETCollection);
      calometcoll->push_back( calomet ) ;
      event.put( calometcoll );           
    }
    //-----------------------------------
    else if( METtype == "TCMET" )
      {
	TCMETAlgo tcmetalgorithm;
	std::auto_ptr<METCollection> tcmetcoll;
	tcmetcoll.reset(new METCollection);
	tcmetcoll->push_back( tcmetalgorithm.CalculateTCMET(event, setup, conf_, &responseFunction_) ) ;
	event.put( tcmetcoll );
      }
    //----------------------------------
    else if( METtype == "PFMET" )
      {
	alg_.run(input, &output, globalThreshold);
	PFSpecificAlgo pf;
	std::auto_ptr<PFMETCollection> pfmetcoll;
	pfmetcoll.reset (new PFMETCollection);
	pfmetcoll->push_back( pf.addInfo(input, output) );
	event.put( pfmetcoll );
      }
    //-----------------------------------
    else if( METtype == "GenMET" ) 
    {
      alg_.run(input, &output, globalThreshold); 
      GenSpecificAlgo gen;
      std::auto_ptr<GenMETCollection> genmetcoll;
      genmetcoll.reset (new GenMETCollection);
      genmetcoll->push_back( gen.addInfo(input, output) );
      event.put( genmetcoll );
    }
    else
    {
      alg_.run(input, &output, globalThreshold); 
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
