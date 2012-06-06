// -*- C++ -*-
//
// Package:    METProducers
// Class:      METProducer
// 
// Original Author:  Rick Cavanaugh
//         Created:  April 4, 2006
// $Id: METProducer.cc,v 1.47 2012/06/06 18:41:37 sakuma Exp $
//
//

//____________________________________________________________________________||
// Modification by R. Remington on 10/21/08
// Added globalThreshold input Parameter to impose on each tower in tower collection
// that is looped over by the CaloSpecificAlgo.  This is in order to fulfill Scheme B threhsolds...   
// Modified:     12.13.2008 by R.Cavanaugh, UIC/Fermilab
// Description:  include Particle Flow MET
// Modified:     12.12.2008  by R. Remington, UFL
// Description:  include TCMET , move alg_.run() inside of relevant if-statements, and add METSignficance algorithm to METtype="CaloMET" cases

//____________________________________________________________________________||
#include "RecoMET/METProducers/interface/METProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/GenMETFwd.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/PFClusterMETFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

#include "RecoMET/METAlgorithms/interface/TCMETAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/PFClusterSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/GenSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/CaloSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignCaloSpecificAlgo.h"

#include <memory>

//____________________________________________________________________________||

// using namespace edm;
// using namespace std;
// using namespace reco;

namespace cms 
{
  METProducer::METProducer(const edm::ParameterSet& iConfig) 
  : alg_(), resolutions_(0), tcmetalgorithm(0) 
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
	produces<reco::CaloMETCollection>().setBranchAlias(alias.c_str()); 
	calculateSignificance_ = iConfig.getParameter<bool>("calculateSignificance");
      }
    else if( METtype == "GenMET" )  
      {
	onlyFiducial = iConfig.getParameter<bool>("onlyFiducialParticles");
        usePt = iConfig.getUntrackedParameter<bool>("usePt", false);
	produces<reco::GenMETCollection>().setBranchAlias(alias.c_str());
      }
    else if( METtype == "PFMET" )
      {
	produces<reco::PFMETCollection>().setBranchAlias(alias.c_str()); 

	calculateSignificance_ = iConfig.getParameter<bool>("calculateSignificance");

	if(calculateSignificance_){
	    jetsLabel_ = iConfig.getParameter<edm::InputTag>("jets");
	}

      }
    else if( METtype == "PFClusterMET" )
      {
	produces<reco::PFClusterMETCollection>().setBranchAlias(alias.c_str()); 
      }
    else if (METtype == "TCMET" )
      {
	produces<reco::METCollection>().setBranchAlias(alias.c_str());

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
      produces<reco::METCollection>().setBranchAlias(alias.c_str()); 

    if (calculateSignificance_ && ( METtype == "CaloMET" || METtype == "PFMET")){
	resolutions_ = new metsig::SignAlgoResolutions(iConfig);
	
    }
  }

  METProducer::METProducer() : alg_() 
  {
    tcmetalgorithm = 0; // why does this constructor exist?
    produces<reco::METCollection>(); 
  }

  METProducer::~METProducer() { delete tcmetalgorithm; }

  void METProducer::produce(edm::Event& event, const edm::EventSetup& setup) 
  {

    edm::Handle<edm::View<reco::Candidate> > input;
    event.getByLabel(inputLabel,input);

    CommonMETData output;

    if( METtype == "CaloMET" ) 
    {
      //Run Basic MET Algorithm
      alg_.run(input, &output, globalThreshold);

      // Run CaloSpecific Algorithm
      CaloSpecificAlgo calospecalgo;
      reco::CaloMET calomet = calospecalgo.addInfo(input,output,noHF, globalThreshold);

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
      std::auto_ptr<reco::CaloMETCollection> calometcoll;
      calometcoll.reset(new reco::CaloMETCollection);
      calometcoll->push_back( calomet ) ;
      event.put( calometcoll );  
      
    }
    //-----------------------------------
    else if( METtype == "TCMET" )
      {
	std::auto_ptr<reco::METCollection> tcmetcoll;
	tcmetcoll.reset(new reco::METCollection);
	tcmetcoll->push_back( tcmetalgorithm->CalculateTCMET(event, setup ) ) ;
	event.put( tcmetcoll );
      }
    //----------------------------------
    else if( METtype == "PFMET" )
      {
	alg_.run(input, &output, globalThreshold);
	PFSpecificAlgo pf;
	std::auto_ptr<reco::PFMETCollection> pfmetcoll;
	pfmetcoll.reset (new reco::PFMETCollection);
	
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
	std::auto_ptr<reco::PFClusterMETCollection> pfclustermetcoll;
	pfclustermetcoll.reset (new reco::PFClusterMETCollection);
	
	pfclustermetcoll->push_back( pfcluster.addInfo(input, output) );
	event.put( pfclustermetcoll );
      }
    //-----------------------------------
    else if( METtype == "GenMET" ) 
    {
      GenSpecificAlgo gen;
      std::auto_ptr<reco::GenMETCollection> genmetcoll;
      genmetcoll.reset (new reco::GenMETCollection);
      genmetcoll->push_back( gen.addInfo(input, &output, globalThreshold, onlyFiducial, usePt) );
      event.put( genmetcoll );
    }
    else
      {
      alg_.run(input, &output, globalThreshold); 

      math::XYZTLorentzVector p4( output.mex, output.mey, 0.0, output.met);
      math::XYZPoint vtx(0,0,0);
      reco::MET met( output.sumet, p4, vtx );
      std::auto_ptr<reco::METCollection> metcoll;
      metcoll.reset(new reco::METCollection);
      metcoll->push_back( met );
      event.put( metcoll );
    }
    //-----------------------------------
  }
  //--------------------------------------------------------------------------
}
