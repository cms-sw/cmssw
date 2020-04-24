// -*- C++ -*-
//
// Package:    EgammaElectronProducers
// Class:      GsfElectronProducer
//
/**\class GsfElectronProducer RecoEgamma/ElectronProducers/src/GsfElectronProducer.cc

 Description: EDProducer of GsfElectron objects

 Implementation:
     <Notes on implementation>
*/

#include "GsfElectronProducer.h"

#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

#include "DataFormats/EcalRecHit/interface/EcalSeverityLevel.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


#include <iostream>

using namespace reco;

/* void GsfElectronProducer::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
 {
  edm::ParameterSetDescription desc ;
  GsfElectronBaseProducer::fillDescription(desc) ;

  // input collections
  desc.add<edm::InputTag>("gsfElectronCoresTag",edm::InputTag("gsfElectronCores")) ;
  desc.add<edm::InputTag>("ecalDrivenGsfElectronsTag",edm::InputTag("ecalDrivenGsfElectrons")) ;
  desc.add<edm::InputTag>("pfMvaTag",edm::InputTag("pfElectronTranslator:pf")) ;

  // steering
  desc.add<bool>("addPflowElectrons",true) ;

  // preselection parameters (tracker driven only electrons)
  desc.add<double>("minSCEtBarrelPflow",0.0) ;
  desc.add<double>("minSCEtEndcapsPflow",0.0) ;
  desc.add<double>("minEOverPBarrelPflow",0.0) ;
  desc.add<double>("maxEOverPBarrelPflow",999999999.) ;
  desc.add<double>("minEOverPEndcapsPflow",0.0) ;
  desc.add<double>("maxEOverPEndcapsPflow",999999999.) ;
  desc.add<double>("maxDeltaEtaBarrelPflow",999999999.) ;
  desc.add<double>("maxDeltaEtaEndcapsPflow",999999999.) ;
  desc.add<double>("maxDeltaPhiBarrelPflow",999999999.) ;
  desc.add<double>("maxDeltaPhiEndcapsPflow",999999999.) ;
  desc.add<double>("hOverEConeSizePflow",0.15) ;
  desc.add<double>("hOverEPtMinPflow",0.) ;
  desc.add<double>("maxHOverEBarrelPflow",999999999.) ;
  desc.add<double>("maxHOverEEndcapsPflow",999999999.) ;
  desc.add<double>("maxHBarrelPflow",0.0) ;
  desc.add<double>("maxHEndcapsPflow",0.0) ;
  desc.add<double>("maxSigmaIetaIetaBarrelPflow",999999999.) ;
  desc.add<double>("maxSigmaIetaIetaEndcapsPflow",999999999.) ;
  desc.add<double>("maxFbremBarrelPflow",999999999.) ;
  desc.add<double>("maxFbremEndcapsPflow",999999999.) ;
  desc.add<bool>("isBarrelPflow",false) ;
  desc.add<bool>("isEndcapsPflow",false) ;
  desc.add<bool>("isFiducialPflow",false) ;
  desc.add<double>("maxTIPPflow",999999999.) ;
  desc.add<double>("minMVAPflow",-0.4) ;

  descriptions.add("produceGsfElectrons",desc) ;
 }
 */
GsfElectronProducer::GsfElectronProducer( const edm::ParameterSet & cfg, const gsfAlgoHelpers::HeavyObjectCache* hoc )
  : GsfElectronBaseProducer(cfg,hoc), pfTranslatorParametersChecked_(false)
 {}

GsfElectronProducer::~GsfElectronProducer()
 {}

void GsfElectronProducer::produce( edm::Event & event, const edm::EventSetup & setup )
 {
  beginEvent(event,setup) ;
  algo_->clonePreviousElectrons() ;
  // don't add pflow only electrons if one so wish
  if (strategyCfg_.addPflowElectrons)
    { algo_->completeElectrons(globalCache()) ; }
  algo_->addPflowInfo() ;
  fillEvent(event) ;
  endEvent() ;
 }

void GsfElectronProducer::beginEvent( edm::Event & event, const edm::EventSetup & setup )
 {
  // extra configuration checks
  if (!pfTranslatorParametersChecked_)
   {
    pfTranslatorParametersChecked_ = true ;
    edm::Handle<edm::ValueMap<float> > pfMva ;
    event.getByToken(inputCfg_.pfMVA,pfMva) ;
    checkPfTranslatorParameters(edm::parameterSet(*pfMva.provenance())) ;
   }

  // call to base class
  GsfElectronBaseProducer::beginEvent(event,setup) ;
 }

void GsfElectronProducer::checkPfTranslatorParameters( edm::ParameterSet const & pset )
 {
  edm::ParameterSet mvaBlock = pset.getParameter<edm::ParameterSet>("MVACutBlock") ;
  double pfTranslatorMinMva = mvaBlock.getParameter<double>("MVACut") ;
  double pfTranslatorUndefined = -99. ;
  if (strategyCfg_.applyPreselection&&(cutsCfgPflow_.minMVA<pfTranslatorMinMva))
   {
    // For pure tracker seeded electrons, if MVA is under translatorMinMva, there is no supercluster
    // of any kind available, so GsfElectronCoreProducer has already discarded the electron.
    edm::LogWarning("GsfElectronAlgo|MvaCutTooLow")
      <<"Parameter minMVAPflow ("<<cutsCfgPflow_.minMVA<<") will have no effect on purely tracker seeded electrons."
      <<" It is inferior to the cut already applied by PFlow translator ("<<pfTranslatorMinMva<<")." ;
   }
  if (strategyCfg_.applyPreselection&&(cutsCfg_.minMVA<pfTranslatorMinMva))
   {
    // For ecal seeded electrons, there is a cluster and GsfElectronCoreProducer has kept all electrons,
    // but when MVA is under translatorMinMva, the translator has not stored the supercluster and
    // forced the MVA value to translatorUndefined
    if (cutsCfg_.minMVA>pfTranslatorUndefined)
     {
      edm::LogWarning("GsfElectronAlgo|IncompletePflowInformation")
        <<"Parameter minMVA  ("<<cutsCfg_.minMVA<<")is inferior to the cut applied by PFlow translator ("<<pfTranslatorMinMva<<")."
        <<" Some ecal (and eventually tracker) seeded electrons may lack their MVA value and PFlow supercluster." ;
     }
    else
     {
      // the MVA value has been forced to translatorUndefined, inferior minMVAPflow
      // so the cut actually applied is the PFlow one
      throw cms::Exception("GsfElectronAlgo|BadMvaCut")
        <<"Parameter minMVA is inferior to the lowest possible value."
        <<" Every electron will be blessed whatever other criteria." ;
     }
   }
 }


