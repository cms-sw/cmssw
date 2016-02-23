// This file is imported from:
//http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/UserCode/Mangano/WWAnalysis/AnalysisStep/plugins/CalibratedPatElectronProducer.cc?revision=1.2&view=markup


// -*- C++ -*-
//
// Package:    EgammaElectronProducers
// Class:      CalibratedPatElectronProducer
//
/**\class CalibratedPatElectronProducer

 Description: EDProducer of PatElectron objects

 Implementation:
     <Notes on implementation>
*/

//#if CMSSW_VERSION>500

#include "EgammaAnalysis/ElectronTools/plugins/CalibratedPatElectronProducer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyCalibrator.h"

#include <iostream>

using namespace edm ;
using namespace std ;
using namespace reco ;
using namespace pat ;

CalibratedPatElectronProducer::CalibratedPatElectronProducer( const edm::ParameterSet & cfg )
// : PatElectronBaseProducer(cfg)
{
    produces<ElectronCollection>();

    inputPatElectronsToken = consumes<edm::View<reco::Candidate> >(cfg.getParameter<edm::InputTag>("inputPatElectronsTag"));
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
    applyExtraHighEnergyProtection = cfg.getParameter<bool>("applyExtraHighEnergyProtection");

    //basic checks
    if ( isMC && ( dataset != "Summer11" && dataset != "Fall11"
        && dataset != "Summer12" && dataset != "Summer12_DR53X_HCP2012"
        && dataset != "Summer12_LegacyPaper" ) )
    {
        throw cms::Exception("CalibratedPATElectronProducer|ConfigError") << "Unknown MC dataset";
    }
    if ( !isMC && ( dataset != "Prompt" && dataset != "ReReco"
        && dataset != "Jan16ReReco" && dataset != "ICHEP2012"
        && dataset != "Moriond2013" && dataset != "22Jan2013ReReco" ) )
    {
        throw cms::Exception("CalibratedPATElectronProducer|ConfigError") << "Unknown Data dataset";
    }

    // Linearity correction only applied on combined momentum obtain with regression combination
    if(combinationType!=3 && applyLinearityCorrection)
    {
        std::cout << "[CalibratedElectronProducer] "
            << "Warning: you chose combinationType!=3 and applyLinearityCorrection=True. Linearity corrections are only applied on top of combination 3." << std::endl;
    }


    std::cout << "[CalibratedPATElectronProducer] Correcting scale for dataset " << dataset << std::endl;

    //initializations
    std::string pathToDataCorr;
    switch ( correctionsType )
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
                throw cms::Exception("CalibratedPATElectronProducer|ConfigError")
                    << "You choose standard non-regression ecal energy scale corrections. They are not implemented yet.";
    	        break;
    	    default:
                throw cms::Exception("CalibratedPATElectronProducer|ConfigError")
                    << "Unknown correctionsType !!!";
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
        std::cout << "[CalibratedPATElectronProducer] "
        << "ElectronEnergyCalibrator object is created " << std::endl;
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
        std::cout << "[CalibratedPATElectronProducer] "
        << "Combination tools are created and initialized " << std::endl;
    }
}



CalibratedPatElectronProducer::~CalibratedPatElectronProducer()
{}

void CalibratedPatElectronProducer::produce( edm::Event & event, const edm::EventSetup & setup )
{

    edm::Handle<edm::View<reco::Candidate> > oldElectrons ;
    event.getByToken(inputPatElectronsToken,oldElectrons) ;
    std::auto_ptr<ElectronCollection> electrons( new ElectronCollection ) ;
    ElectronCollection::const_iterator electron ;
    ElectronCollection::iterator ele ;
    // first clone the initial collection
    for
        (
            edm::View<reco::Candidate>::const_iterator ele=oldElectrons->begin();
            ele!=oldElectrons->end();
            ++ele
        )
    {
        const pat::ElectronRef elecsRef = edm::RefToBase<reco::Candidate>(oldElectrons,ele-oldElectrons->begin()).castTo<pat::ElectronRef>();
        pat::Electron clone = *edm::RefToBase<reco::Candidate>(oldElectrons,ele-oldElectrons->begin()).castTo<pat::ElectronRef>();
        electrons->push_back(clone);
    }

    if (correctionsType != 0 )
    {
        for
            (
                ele = electrons->begin();
                ele != electrons->end() ;
                ++ele
            )
        {
            int elClass = -1;
            int run = event.run();

            float r9 = ele->r9();
            double correctedEcalEnergy = ele->correctedEcalEnergy();
            double correctedEcalEnergyError = ele->correctedEcalEnergyError();
            double trackMomentum = ele->trackMomentumAtVtx().R();
            double trackMomentumError = ele->trackMomentumError();
            double combinedMomentum = ele->p();
            double combinedMomentumError = 0;
            if ( ele->candidateP4Kind() != GsfElectron::P4_UNKNOWN )
            {
              combinedMomentumError = ele->p4Error(ele->candidateP4Kind());
            }
            // FIXME : p4Error not filled for pure tracker electrons
            // Recompute it using the parametrization implemented in
            // RecoEgamma/EgammaElectronAlgos/src/ElectronEnergyCorrector.cc::simpleParameterizationUncertainty()
            if( !ele->ecalDrivenSeed() )
            {
                double error = 999. ;
                double momentum = (combinedMomentum<15. ? 15. : combinedMomentum);
                if ( ele->isEB() )
                {
                    float parEB[3] = { 5.24e-02,  2.01e-01, 1.00e-02} ;
                    error = momentum * sqrt( pow(parEB[0]/sqrt(momentum),2) + pow(parEB[1]/momentum,2) + pow(parEB[2],2) );
                }
                else if ( ele->isEE() )
                {
                    float parEE[3] = { 1.46e-01, 9.21e-01, 1.94e-03} ;
                    error = momentum * sqrt( pow(parEE[0]/sqrt(momentum),2) + pow(parEE[1]/momentum,2) + pow(parEE[2],2) );
                }
                combinedMomentumError = error;
            }

            if (ele->classification() == reco::GsfElectron::GOLDEN) {elClass = 0;}
            if (ele->classification() == reco::GsfElectron::BIGBREM) {elClass = 1;}
            if (ele->classification() == reco::GsfElectron::BADTRACK) {elClass = 2;}
            if (ele->classification() == reco::GsfElectron::SHOWERING) {elClass = 3;}
            if (ele->classification() == reco::GsfElectron::GAP) {elClass = 4;}

            SimpleElectron mySimpleElectron
                (
                    run,
                    elClass,
                    r9,
                    correctedEcalEnergy,
                    correctedEcalEnergyError,
                    trackMomentum,
                    trackMomentumError,
                    ele->ecalRegressionEnergy(),
                    ele->ecalRegressionError(),
                    combinedMomentum,
                    combinedMomentumError,
                    ele->superCluster()->eta(),
                    ele->isEB(),
                    isMC,
                    ele->ecalDriven(),
                    ele->trackerDrivenSeed()
                );

            // energy calibration for ecalDriven electrons
            if ( ele->core()->ecalDrivenSeed() || correctionsType==2 || combinationType==3 )
            {
                theEnCorrector->calibrate(mySimpleElectron, event.streamID());

                // E-p combination

                switch ( combinationType )
                {
              	    case 0:
                		if ( verbose )
                        {
                            std::cout << "[CalibratedPATElectronProducer] "
                            << "You choose not to combine." << std::endl;
                        }
                		break;
                	case 1:
                	    if ( verbose )
                        {
                            std::cout << "[CalibratedPATElectronProducer] "
                            << "You choose corrected regression energy for standard combination" << std::endl;
                        }
                	    myCombinator->setCombinationMode(1);
                	    myCombinator->combine(mySimpleElectron);
                	    break;
                	case 2:
                	    if ( verbose )
                        {
                            std::cout << "[CalibratedPATElectronProducer] "
                            << "You choose uncorrected regression energy for standard combination" << std::endl;
                        }
                	    myCombinator->setCombinationMode(2);
                	    myCombinator->combine(mySimpleElectron);
                	    break;
                	case 3:
                	    if ( verbose )
                        {
                            std::cout << "[CalibratedPATElectronProducer] "
                            << "You choose regression combination." << std::endl;
                        }
                	    myEpCombinationTool->combine(mySimpleElectron, applyExtraHighEnergyProtection);
                        theEnCorrector->correctLinearity(mySimpleElectron);
                	    break;
                	default:
                		  throw cms::Exception("CalibratedPATElectronProducer|ConfigError")
                              << "Unknown combination Type !!!" ;
                }

                math::XYZTLorentzVector oldMomentum = ele->p4() ;
                math::XYZTLorentzVector newMomentum_ ;
                newMomentum_ = math::XYZTLorentzVector
                 ( oldMomentum.x()*mySimpleElectron.getCombinedMomentum()/oldMomentum.t(),
                   oldMomentum.y()*mySimpleElectron.getCombinedMomentum()/oldMomentum.t(),
                   oldMomentum.z()*mySimpleElectron.getCombinedMomentum()/oldMomentum.t(),
                   mySimpleElectron.getCombinedMomentum() ) ;

                 ele->correctMomentum
                    (
                        newMomentum_,
                        mySimpleElectron.getTrackerMomentumError(),
                        mySimpleElectron.getCombinedMomentumError()
                    );

                if ( verbose )
                {
                    std::cout << "[CalibratedPATElectronProducer] Combined momentum after saving  "
                        << ele->p4().t() << std::endl;
                }
            }// end of if (ele.core()->ecalDrivenSeed())
        }// end of loop on electrons
    } else
    {
        if ( verbose )
        {
            std::cout << "[CalibratedPATElectronProducer] "
            << "You choose not to correct. Uncorrected Regression Energy is taken." << std::endl;
        }
    }
    // Save the electrons
    event.put(electrons) ;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_MODULE(CalibratedPatElectronProducer);
