// -*- C++ -*-
//
// Package:    METProducers
// Class:      METProducer
// 
// Original Author:  Rick Cavanaugh
//         Created:  April 4, 2006
// $Id: METProducer.cc,v 1.53 2013/05/07 13:16:16 salee Exp $
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

#include "RecoMET/METAlgorithms/interface/METAlgo.h" 
#include "RecoMET/METAlgorithms/interface/TCMETAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/PFClusterSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/GenSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/CaloSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignCaloSpecificAlgo.h"

#include <memory>

//____________________________________________________________________________||
namespace cms 
{
  METProducer::METProducer(const edm::ParameterSet& iConfig) 
    : inputLabel(iConfig.getParameter<edm::InputTag>("src"))
    , inputType(iConfig.getParameter<std::string>("InputType"))
    , METtype(iConfig.getParameter<std::string>("METType"))
    , alias(iConfig.getParameter<std::string>("alias"))
    , calculateSignificance_(false)
    , resolutions_(0)
    , globalThreshold(iConfig.getParameter<double>("globalThreshold"))
  {
    if( METtype == "CaloMET" ) 
      {
	noHF = iConfig.getParameter<bool>("noHF");
	produces<reco::CaloMETCollection>().setBranchAlias(alias.c_str()); 
	calculateSignificance_ = iConfig.getParameter<bool>("calculateSignificance");
      }
    else if( METtype == "GenMET" )  
      {
	onlyFiducial = iConfig.getParameter<bool>("onlyFiducialParticles");
        usePt = iConfig.getParameter<bool>("usePt");
        applyFiducialThresholdForFractions = iConfig.getParameter<bool>("applyFiducialThresholdForFractions");
	produces<reco::GenMETCollection>().setBranchAlias(alias.c_str());
      }
    else if( METtype == "PFMET" )
      {
	produces<reco::PFMETCollection>().setBranchAlias(alias.c_str()); 

	calculateSignificance_ = iConfig.getParameter<bool>("calculateSignificance");

	if(calculateSignificance_)
	  {
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

	int responseFunctionType = 0;
	if(! correctShowerTracks_)
	  {
	    if( rfType_ == 1 ) responseFunctionType = 1; // 'fit'
	    else if( rfType_ == 2 ) responseFunctionType = 2; // 'mode'
	    else { /* probably error */ }
	  }
	tcMetAlgo_.configure(iConfig, responseFunctionType );
      }
    else                            
      produces<reco::METCollection>().setBranchAlias(alias.c_str()); 

    if (calculateSignificance_ && ( METtype == "CaloMET" || METtype == "PFMET")){
	resolutions_ = new metsig::SignAlgoResolutions(iConfig);
	
    }
  }


  void METProducer::produce(edm::Event& event, const edm::EventSetup& setup) 
  {
    if( METtype == "CaloMET" ) 
      {
	produce_CaloMET(event);
	return;
      }

    if( METtype == "TCMET" )
      {
	produce_TCMET(event, setup);
	return;
      }

    if( METtype == "PFMET" )
      {
	produce_PFMET(event);
	return;
      }

    if( METtype == "PFClusterMET" )
      {
	produce_PFClusterMET(event);
	return;
      }

    if( METtype == "GenMET" ) 
      {
	produce_GenMET(event);
	return;
      }

    produce_else(event);
  }

  void METProducer::produce_CaloMET(edm::Event& event)
  {
    edm::Handle<edm::View<reco::Candidate> > input;
    event.getByLabel(inputLabel, input);

    METAlgo algo;
    CommonMETData commonMETdata = algo.run(input, globalThreshold);

    CaloSpecificAlgo calospecalgo;
    reco::CaloMET calomet = calospecalgo.addInfo(input, commonMETdata, noHF, globalThreshold);

    if( calculateSignificance_ ) 
      {
	SignCaloSpecificAlgo signcalospecalgo;
	signcalospecalgo.calculateBaseCaloMET(input, commonMETdata, *resolutions_, noHF, globalThreshold);
	calomet.SetMetSignificance(signcalospecalgo.getSignificance() );
	calomet.setSignificanceMatrix(signcalospecalgo.getSignificanceMatrix());
      }

    std::auto_ptr<reco::CaloMETCollection> calometcoll;
    calometcoll.reset(new reco::CaloMETCollection);
    calometcoll->push_back( calomet ) ;
    event.put( calometcoll );  
  }

  void METProducer::produce_TCMET(edm::Event& event, const edm::EventSetup& setup)
  {
    std::auto_ptr<reco::METCollection> tcmetcoll;
    tcmetcoll.reset(new reco::METCollection);
    tcmetcoll->push_back( tcMetAlgo_.CalculateTCMET(event, setup ) ) ;
    event.put( tcmetcoll );
  }

  void METProducer::produce_PFMET(edm::Event& event)
  {
    edm::Handle<edm::View<reco::Candidate> > input;
    event.getByLabel(inputLabel, input);

    METAlgo algo;
    CommonMETData commonMETdata = algo.run(input, globalThreshold);

    PFSpecificAlgo pf;
	
    if( calculateSignificance_ )
      {
	edm::Handle<edm::View<reco::PFJet> > jets;
	event.getByLabel(jetsLabel_, jets);
	pf.runSignificance(*resolutions_, jets);
      }

    std::auto_ptr<reco::PFMETCollection> pfmetcoll;
    pfmetcoll.reset(new reco::PFMETCollection);
    pfmetcoll->push_back( pf.addInfo(input, commonMETdata) );
    event.put( pfmetcoll );
  }

  void METProducer::produce_PFClusterMET(edm::Event& event)
  {
    edm::Handle<edm::View<reco::Candidate> > input;
    event.getByLabel(inputLabel, input);

    METAlgo algo;
    CommonMETData commonMETdata = algo.run(input, globalThreshold);

    PFClusterSpecificAlgo pfcluster;
    std::auto_ptr<reco::PFClusterMETCollection> pfclustermetcoll;
    pfclustermetcoll.reset (new reco::PFClusterMETCollection);
	
    pfclustermetcoll->push_back( pfcluster.addInfo(input, commonMETdata) );
    event.put( pfclustermetcoll );
  }

  void METProducer::produce_GenMET(edm::Event& event)
  {
    edm::Handle<edm::View<reco::Candidate> > input;
    event.getByLabel(inputLabel, input);

    CommonMETData commonMETdata;

    GenSpecificAlgo gen;
    std::auto_ptr<reco::GenMETCollection> genmetcoll;
    genmetcoll.reset (new reco::GenMETCollection);
    genmetcoll->push_back( gen.addInfo(input, &commonMETdata, globalThreshold, onlyFiducial,applyFiducialThresholdForFractions, usePt) );
    event.put( genmetcoll );
  }
  
  void METProducer::produce_else(edm::Event& event)
  {
    edm::Handle<edm::View<reco::Candidate> > input;
    event.getByLabel(inputLabel, input);

    CommonMETData commonMETdata;

    METAlgo algo;
    algo.run(input, &commonMETdata, globalThreshold); 

    math::XYZTLorentzVector p4( commonMETdata.mex, commonMETdata.mey, 0.0, commonMETdata.met);
    math::XYZPoint vtx(0,0,0);
    reco::MET met( commonMETdata.sumet, p4, vtx );
    std::auto_ptr<reco::METCollection> metcoll;
    metcoll.reset(new reco::METCollection);
    metcoll->push_back( met );
    event.put( metcoll );
  }
    
}
