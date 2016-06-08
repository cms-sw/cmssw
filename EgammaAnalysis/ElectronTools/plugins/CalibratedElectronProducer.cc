// This file is imported from:

// -*- C++ -*-
//
// Package:    EgammaElectronProducers
// Class:      CalibratedElectronProducer
//
/**\class CalibratedElectronProducer

 Description: EDProducer of GsfElectron objects

 Implementation:
     <Notes on implementation>
*/

#include "EgammaAnalysis/ElectronTools/plugins/CalibratedElectronProducer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "EgammaAnalysis/ElectronTools/interface/SuperClusterHelper.h"

#include <iostream>

CalibratedElectronProducer::CalibratedElectronProducer( const edm::ParameterSet & cfg )
{
  inputElectronsToken_ = consumes<reco::GsfElectronCollection>(cfg.getParameter<edm::InputTag>("inputElectronsTag"));
  
  energyRegToken_      = consumes<edm::ValueMap<double> >(cfg.getParameter<edm::InputTag>("nameEnergyReg"));
  energyErrorRegToken_ = consumes<edm::ValueMap<double> >(cfg.getParameter<edm::InputTag>("nameEnergyErrorReg"));
  
  recHitCollectionEBToken_ = consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitCollectionEB"));
  recHitCollectionEEToken_ = consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitCollectionEE"));

  
  nameNewEnergyReg_      = cfg.getParameter<std::string>("nameNewEnergyReg");
  nameNewEnergyErrorReg_ = cfg.getParameter<std::string>("nameNewEnergyErrorReg");
  newElectronName_ = cfg.getParameter<std::string>("outputGsfElectronCollectionLabel");


  dataset = cfg.getParameter<std::string>("inputDataset");
  isMC = cfg.getParameter<bool>("isMC");
  updateEnergyError = cfg.getParameter<bool>("updateEnergyError");
  lumiRatio = cfg.getParameter<double>("lumiRatio");
  correctionsType = cfg.getParameter<int>("correctionsType");
  applyLinearityCorrection = cfg.getParameter<bool>("applyLinearityCorrection");
  combinationType = cfg.getParameter<int>("combinationType");
  verbose = cfg.getParameter<bool>("verbose");
  synchronization = cfg.getParameter<bool>("synchronization");
  combinationRegressionInputPath = cfg.getParameter<std::string>("combinationRegressionInputPath");
  scaleCorrectionsInputPath = cfg.getParameter<std::string>("scaleCorrectionsInputPath");
  linCorrectionsInputPath   = cfg.getParameter<std::string>("linearityCorrectionsInputPath");
  
  //basic checks
  if ( isMC && ( dataset != "Summer11" && dataset != "Fall11"
		 && dataset!= "Summer12" && dataset != "Summer12_DR53X_HCP2012"
		 && dataset != "Summer12_LegacyPaper" ) )
    {
      throw cms::Exception("CalibratedgsfElectronProducer|ConfigError") << "Unknown MC dataset";
    }
    if ( !isMC && ( dataset != "Prompt" && dataset != "ReReco"
		    && dataset != "Jan16ReReco" && dataset != "ICHEP2012"
		    && dataset != "Moriond2013" && dataset != "22Jan2013ReReco" ) )
      {
        throw cms::Exception("CalibratedgsfElectronProducer|ConfigError") << "Unknown Data dataset";
    }
    
    // Linearity correction only applied on combined momentum obtain with regression combination
    if(combinationType!=3 && applyLinearityCorrection)
      {
        std::cout << "[CalibratedElectronProducer] "
		  << "Warning: you chose combinationType!=3 and applyLinearityCorrection=True. Linearity corrections are only applied on top of combination 3." << std::endl;
      }
    
    std::cout << "[CalibratedGsfElectronProducer] Correcting scale for dataset " << dataset << std::endl;
    
    //initializations
    std::string pathToDataCorr;
    switch (correctionsType)
      {
      case 0:
	break;
      case 1:
	if ( verbose )
	  {
	    std::cout << "You choose regression 1 scale corrections" << std::endl;
	  }
	break;
      case 2:
	if ( verbose )
	  {
	    std::cout << "You choose regression 2 scale corrections." << std::endl;
	  }
	break;
      case 3:
	throw cms::Exception("CalibratedgsfElectronProducer|ConfigError")
	  << "You choose standard non-regression ecal energy scale corrections. They are not implemented yet.";
	break;
      default:
	throw cms::Exception("CalibratedgsfElectronProducer|ConfigError")
	  << "Unknown correctionsType !!!" ;
      }
    
    theEnCorrector = new ElectronEnergyCalibrator
      (
       edm::FileInPath(scaleCorrectionsInputPath.c_str()).fullPath().c_str(),
       edm::FileInPath(linCorrectionsInputPath.c_str()).fullPath().c_str(),
       dataset,
       correctionsType,
       applyLinearityCorrection,
       lumiRatio,
       isMC,
       updateEnergyError,
       verbose,
       synchronization
       );
    
    if ( verbose )
      {
        std::cout<<"[CalibratedGsfElectronProducer] "
		 << "ElectronEnergyCalibrator object is created" << std::endl;
      }
    
    myEpCombinationTool = new EpCombinationTool();
    myEpCombinationTool->init
      (
       edm::FileInPath(combinationRegressionInputPath.c_str()).fullPath().c_str(),
       "CombinationWeight"
       );
    
    myCombinator = new ElectronEPcombinator();
    
    if ( verbose )
      {
        std::cout << "[CalibratedGsfElectronProducer] "
		  << "Combination tools are created and initialized" << std::endl;
      }
    
    produces<edm::ValueMap<double> >(nameNewEnergyReg_);
    produces<edm::ValueMap<double> >(nameNewEnergyErrorReg_);
    produces<reco::GsfElectronCollection> (newElectronName_);
    geomInitialized_ = false;
}

CalibratedElectronProducer::~CalibratedElectronProducer()
{}

void CalibratedElectronProducer::produce( edm::Event & event, const edm::EventSetup & setup )
{
  if (!geomInitialized_)
    {
      edm::ESHandle<CaloTopology> theCaloTopology;
      setup.get<CaloTopologyRecord>().get(theCaloTopology);
      ecalTopology_ = & (*theCaloTopology);
      
      edm::ESHandle<CaloGeometry> theCaloGeometry;
      setup.get<CaloGeometryRecord>().get(theCaloGeometry);
      caloGeometry_ = & (*theCaloGeometry);
      geomInitialized_ = true;
    }
  
  // Read GsfElectrons
  edm::Handle<reco::GsfElectronCollection>  oldElectronsH ;
  event.getByToken(inputElectronsToken_,oldElectronsH) ;
  
  // Read RecHits
  edm::Handle< EcalRecHitCollection > pEBRecHits;
  edm::Handle< EcalRecHitCollection > pEERecHits;
  event.getByToken( recHitCollectionEBToken_, pEBRecHits );
  event.getByToken( recHitCollectionEEToken_, pEERecHits );

  // ReadValueMaps
  edm::Handle<edm::ValueMap<double> > valMapEnergyH;
  event.getByToken(energyRegToken_,valMapEnergyH);
  edm::Handle<edm::ValueMap<double> > valMapEnergyErrorH;
  event.getByToken(energyErrorRegToken_,valMapEnergyErrorH);
  
  // Prepare output collections
  std::auto_ptr<reco::GsfElectronCollection> electrons( new reco::GsfElectronCollection ) ;
  // Fillers for ValueMaps:
  std::auto_ptr<edm::ValueMap<double> > regrNewEnergyMap(new edm::ValueMap<double>() );
  edm::ValueMap<double>::Filler energyFiller(*regrNewEnergyMap);
  
  std::auto_ptr<edm::ValueMap<double> > regrNewEnergyErrorMap(new edm::ValueMap<double>() );
  edm::ValueMap<double>::Filler energyErrorFiller(*regrNewEnergyErrorMap);
  
  // first clone the initial collection
  unsigned nElectrons = oldElectronsH->size();
  for( unsigned iele = 0; iele < nElectrons; ++iele )
    {
      electrons->push_back((*oldElectronsH)[iele]);
    }
  
  std::vector<double> regressionValues;
  std::vector<double> regressionErrorValues;
  regressionValues.reserve(nElectrons);
  regressionErrorValues.reserve(nElectrons);
  
  if ( correctionsType != 0 )
    {
      for ( unsigned iele = 0; iele < nElectrons ; ++iele)
        {
	  reco::GsfElectron & ele  ( (*electrons)[iele]);
	  reco::GsfElectronRef elecRef(oldElectronsH,iele);
	  double regressionEnergy = (*valMapEnergyH)[elecRef];
	  double regressionEnergyError = (*valMapEnergyErrorH)[elecRef];
	  
	  regressionValues.push_back(regressionEnergy);
	  regressionErrorValues.push_back(regressionEnergyError);
	  
	  //    r9
	  const EcalRecHitCollection * recHits=0;
	  if( ele.isEB() )
            {
	      recHits = pEBRecHits.product();
            } else recHits = pEERecHits.product();
	  
	  SuperClusterHelper mySCHelper( &(ele), recHits, ecalTopology_, caloGeometry_ );
	  
	  int elClass = -1;
	  int run = event.run();
	  
	  float r9 = mySCHelper.r9();
	  double correctedEcalEnergy = ele.correctedEcalEnergy();
	  double correctedEcalEnergyError = ele.correctedEcalEnergyError();
	  double trackMomentum = ele.trackMomentumAtVtx().R();
	  double trackMomentumError = ele.trackMomentumError();
	  double combinedMomentum = ele.p();
	  double combinedMomentumError = ele.p4Error(ele.candidateP4Kind());
	  // FIXME : p4Error not filled for pure tracker electrons
	  // Recompute it using the parametrization implemented in
	  // RecoEgamma/EgammaElectronAlgos/src/ElectronEnergyCorrector.cc::simpleParameterizationUncertainty()
	  if( !ele.ecalDrivenSeed() )
            {
	      double error = 999. ;
	      double momentum = (combinedMomentum<15. ? 15. : combinedMomentum);
	      if ( ele.isEB() )
                {
		  float parEB[3] = { 5.24e-02,  2.01e-01, 1.00e-02};
		  error = momentum * sqrt( pow(parEB[0]/sqrt(momentum),2) + pow(parEB[1]/momentum,2) + pow(parEB[2],2) );
                }
	      else if ( ele.isEE() )
                {
		  float parEE[3] = { 1.46e-01, 9.21e-01, 1.94e-03} ;
		  error = momentum * sqrt( pow(parEE[0]/sqrt(momentum),2) + pow(parEE[1]/momentum,2) + pow(parEE[2],2) );
                }
	      combinedMomentumError = error;
            }
	  
	  if (ele.classification() == reco::GsfElectron::GOLDEN) {elClass = 0;}
	  if (ele.classification() == reco::GsfElectron::BIGBREM) {elClass = 1;}
	  if (ele.classification() == reco::GsfElectron::BADTRACK) {elClass = 2;}
	  if (ele.classification() == reco::GsfElectron::SHOWERING) {elClass = 3;}
	  if (ele.classification() == reco::GsfElectron::GAP) {elClass = 4;}
	  
	  SimpleElectron mySimpleElectron
	    (
	     run,
	     elClass,
	     r9,
	     correctedEcalEnergy,
	     correctedEcalEnergyError,
	     trackMomentum,
	     trackMomentumError,
	     regressionEnergy,
	     regressionEnergyError,
	     combinedMomentum,
	     combinedMomentumError,
	     ele.superCluster()->eta(),
	     ele.isEB(),
	     isMC,
	     ele.ecalDriven(),
	     ele.trackerDrivenSeed()
	     );
	  
	  // energy calibration for ecalDriven electrons
	  if ( ele.core()->ecalDrivenSeed() || correctionsType==2 || combinationType==3 )
            {
	      theEnCorrector->calibrate(mySimpleElectron, event.streamID());
	      
	      // E-p combination
	      
	      switch ( combinationType )
                {
		case 0:
		  if ( verbose )
		    {
		      std::cout << "[CalibratedGsfElectronProducer] "
				<< "You choose not to combine." << std::endl;
		    }
		  break;
		case 1:
		  if ( verbose )
		    {
		      std::cout << "[CalibratedGsfElectronProducer] "
				<< "You choose corrected regression energy for standard combination" << std::endl;
		    }
		  myCombinator->setCombinationMode(1);
		  myCombinator->combine(mySimpleElectron);
		  break;
		case 2:
		  if ( verbose )
		    {
		      std::cout << "[CalibratedGsfElectronProducer] "
				<< "You choose uncorrected regression energy for standard combination" << std::endl;
		    }
		  myCombinator->setCombinationMode(2);
		  myCombinator->combine(mySimpleElectron);
		  break;
		case 3:
		  if ( verbose )
		    {
		      std::cout << "[CalibratedGsfElectronProducer] "
				<< "You choose regression combination." << std::endl;
		    }
		  myEpCombinationTool->combine(mySimpleElectron);
		  theEnCorrector->correctLinearity(mySimpleElectron);
		  break;
		default:
		  throw cms::Exception("CalibratedgsfElectronProducer|ConfigError")
		    << "Unknown combination Type !!!" ;
                }
	      
	      math::XYZTLorentzVector oldMomentum = ele.p4() ;
	      math::XYZTLorentzVector newMomentum_ ;
	      newMomentum_ = math::XYZTLorentzVector
		( oldMomentum.x()*mySimpleElectron.getCombinedMomentum()/oldMomentum.t(),
		  oldMomentum.y()*mySimpleElectron.getCombinedMomentum()/oldMomentum.t(),
		  oldMomentum.z()*mySimpleElectron.getCombinedMomentum()/oldMomentum.t(),
		  mySimpleElectron.getCombinedMomentum() ) ;
	      
	      ele.correctMomentum
		(
		 newMomentum_,
		 mySimpleElectron.getTrackerMomentumError(),
		 mySimpleElectron.getCombinedMomentumError()
		 );
	      
	      if ( verbose )
                {
		  std::cout << "[CalibratedGsfElectronProducer] Combined momentum after saving "
			    << ele.p4().t() << std::endl;
                }
            }// end of if (ele.core()->ecalDrivenSeed())
        }// end of loop on electrons
    } else
    {
      if ( verbose )
        {
	  std::cout << "[CalibratedGsfElectronProducer] "
		    << "You choose not to correct. Uncorrected Regression Energy is taken." << std::endl;
        }
    }
  
  // Save the electrons
  const edm::OrphanHandle<reco::GsfElectronCollection> gsfNewElectronHandle = event.put(electrons, newElectronName_) ;
  energyFiller.insert(gsfNewElectronHandle,regressionValues.begin(),regressionValues.end());
  energyFiller.fill();
  energyErrorFiller.insert(gsfNewElectronHandle,regressionErrorValues.begin(),regressionErrorValues.end());
  energyErrorFiller.fill();
  
  event.put(regrNewEnergyMap,nameNewEnergyReg_);
  event.put(regrNewEnergyErrorMap,nameNewEnergyErrorReg_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_MODULE(CalibratedElectronProducer);
