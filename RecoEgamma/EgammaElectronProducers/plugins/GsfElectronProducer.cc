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
