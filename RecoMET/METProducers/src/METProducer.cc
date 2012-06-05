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
#include "RecoMET/METAlgorithms/interface/PFClusterSpecificAlgo.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
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
  METProducer::METProducer(const edm::ParameterSet& iConfig) : alg_() , resolutions_(0), tcmetalgorithm(0) 
  {
    inputLabel = iConfig.getParameter<edm::InputTag>("src");
    inputType  = iConfig.getParameter<std::string>("InputType");
    METtype    = iConfig.getParameter<std::string>("METType");
    alias      = iConfig.getParameter<std::string>("alias");
    globalThreshold = iConfig.getParameter<double>("globalThreshold");
    calculateSignificance_ = false ;

    if( METtype == "CaloMET" ) 
      {
	noHF = iConfig.getParameter<bool>("noHF");
	produces<CaloMETCollection>().setBranchAlias(alias.c_str()); 
	calculateSignificance_ = iConfig.getParameter<bool>("calculateSignificance");
      }
    else if( METtype == "GenMET" )  
      {
	onlyFiducial = iConfig.getParameter<bool>("onlyFiducialParticles");
        usePt      = iConfig.getUntrackedParameter<bool>("usePt",false);
	produces<GenMETCollection>().setBranchAlias(alias.c_str());
      }
    else if( METtype == "PFMET" )
      {
	produces<PFMETCollection>().setBranchAlias(alias.c_str()); 

	calculateSignificance_ = iConfig.getParameter<bool>("calculateSignificance");

	if(calculateSignificance_){
	    jetsLabel_ = iConfig.getParameter<edm::InputTag>("jets");
	}

      }
    else if( METtype == "PFClusterMET" )
      {
	produces<PFClusterMETCollection>().setBranchAlias(alias.c_str()); 
      }
    else if (METtype == "TCMET" )
      {
	produces<METCollection>().setBranchAlias(alias.c_str());

	int rfType_               = iConfig.getParameter<int>("rf_type");
	bool correctShowerTracks_ = iConfig.getParameter<bool>("correctShowerTracks"); 

	if(correctShowerTracks_){
          // use 'shower' and 'noshower' response functions
          myResponseFunctionType = 0;
      	}else{
	  
	  if( rfType_ == 1 ){
            // use response function 'fit'
            myResponseFunctionType = 1;
          }
	  else if( rfType_ == 2 ){
            // use response function 'mode'
            myResponseFunctionType = 2;
          }
        }
        tcmetalgorithm = new TCMETAlgo();
	tcmetalgorithm->configure(iConfig, myResponseFunctionType );
      }
    else                            
      produces<METCollection>().setBranchAlias(alias.c_str()); 

    if (calculateSignificance_ && ( METtype == "CaloMET" || METtype == "PFMET")){
	resolutions_ = new metsig::SignAlgoResolutions(iConfig);
	
    }
  }
  //--------------------------------------------------------------------------

  //--------------------------------------------------------------------------
  // Default Constructor
  //-----------------------------------
  METProducer::METProducer() : alg_() 
  {
    tcmetalgorithm = 0; // why does this constructor exist?
    produces<METCollection>(); 
  }
  //--------------------------------------------------------------------------

  //--------------------------------------------------------------------------
  // Default Destructor
  //-----------------------------------
  METProducer::~METProducer() { delete tcmetalgorithm;}
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
      //Run Basic MET Algorithm
      alg_.run(input, &output, globalThreshold);

      // Run CaloSpecific Algorithm
      CaloSpecificAlgo calospecalgo;
      CaloMET calomet = calospecalgo.addInfo(input,output,noHF, globalThreshold);

      //Run algorithm to calculate CaloMET Significance and add to the MET Object
      if( calculateSignificance_ ) 
      {
	  SignCaloSpecificAlgo signcalospecalgo;
	  //metsig::SignAlgoResolutions resolutions(conf_);
	  
	  signcalospecalgo.calculateBaseCaloMET(input,output,*resolutions_,noHF,globalThreshold);
	  calomet.SetMetSignificance( signcalospecalgo.getSignificance() );
	  calomet.setSignificanceMatrix(signcalospecalgo.getSignificanceMatrix());
	}
      //Store CaloMET object in CaloMET collection 
      std::auto_ptr<CaloMETCollection> calometcoll;
      calometcoll.reset(new CaloMETCollection);
      calometcoll->push_back( calomet ) ;
      event.put( calometcoll );  
      
    }
    //-----------------------------------
    else if( METtype == "TCMET" )
      {
	std::auto_ptr<METCollection> tcmetcoll;
	tcmetcoll.reset(new METCollection);
	tcmetcoll->push_back( tcmetalgorithm->CalculateTCMET(event, setup ) ) ;
	event.put( tcmetcoll );
      }
    //----------------------------------
    else if( METtype == "PFMET" )
      {
	alg_.run(input, &output, globalThreshold);
	PFSpecificAlgo pf;
	std::auto_ptr<PFMETCollection> pfmetcoll;
	pfmetcoll.reset (new PFMETCollection);
	
	// add resolutions and calculate significance
	if( calculateSignificance_ )
	  {
	    //metsig::SignAlgoResolutions resolutions(conf_);
	    edm::Handle<edm::View<reco::PFJet> > jets;
	    event.getByLabel(jetsLabel_,jets);
	    pf.runSignificance(*resolutions_, jets);
	  }
	pfmetcoll->push_back( pf.addInfo(input, output) );
	event.put( pfmetcoll );
      }
    //----------------------------------
    else if( METtype == "PFClusterMET" )
      {
	alg_.run(input, &output, globalThreshold);
	PFClusterSpecificAlgo pfcluster;
	std::auto_ptr<PFClusterMETCollection> pfclustermetcoll;
	pfclustermetcoll.reset (new PFClusterMETCollection);
	
	pfclustermetcoll->push_back( pfcluster.addInfo(input, output) );
	event.put( pfclustermetcoll );
      }
    //-----------------------------------
    else if( METtype == "GenMET" ) 
    {
      GenSpecificAlgo gen;
      std::auto_ptr<GenMETCollection> genmetcoll;
      genmetcoll.reset (new GenMETCollection);
      genmetcoll->push_back( gen.addInfo(input, &output, globalThreshold, onlyFiducial, usePt) );
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
