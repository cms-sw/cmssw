#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/EvtGenInterface/interface/EvtGenFactory.h"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenInterfaceBase.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaFactory.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaInterfaceBase.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosFactory.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosInterfaceBase.h"

#include "HepMC/GenEvent.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace gen;
using namespace edm;

CLHEP::HepRandomEngine* decayRandomEngine;

ExternalDecayDriver::ExternalDecayDriver( const ParameterSet& pset )
  : hastauolapp(false),
     fIsInitialized(false),
     fTauolaInterface(0),
     fEvtGenInterface(0),
     fPhotosInterface(0)
{
  
  std::vector<std::string> extGenNames = pset.getParameter< std::vector<std::string> >("parameterSets");
  
  Service<RandomNumberGenerator> rng;
  if(!rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
      "which appears to be absent.  Please add that service to your configuration\n"
      "or remove the modules that require it." << std::endl;
  } 
  decayRandomEngine = &rng->getEngine();   
  
  for (unsigned int ip=0; ip<extGenNames.size(); ++ip ){
    std::string curSet = extGenNames[ip];
    if ( curSet == "EvtGen" || curSet == "EvtGenLHC91"){
      fEvtGenInterface = (EvtGenInterfaceBase*)(EvtGenFactory::get()->create("EvtGenLHC91", pset.getUntrackedParameter< ParameterSet >(curSet)));
      fEvtGenInterface->SetPhotosDecayRandomEngine(decayRandomEngine);
    }
    if ( curSet == "Tauola" || curSet == "Tauola271215" ){
      fTauolaInterface = (TauolaInterfaceBase*)(TauolaFactory::get()->create("Tauola271215", pset.getUntrackedParameter< ParameterSet >(curSet)));
      fTauolaInterface->SetDecayRandomEngine(decayRandomEngine);
    }
    if(curSet =="Tauolapp111a"){
      fTauolaInterface = (TauolaInterfaceBase*)(TauolaFactory::get()->create("Tauolapp111a", pset.getUntrackedParameter< ParameterSet >(curSet)));
      fPhotosInterface = (PhotosInterfaceBase*)(PhotosFactory::get()->create("Photos2155", pset.getUntrackedParameter< ParameterSet >(curSet)));
      fPhotosInterface->SetDecayRandomEngine(decayRandomEngine);
      fPhotosInterface->configureOnlyFor(15); 
      fPhotosInterface->avoidTauLeptonicDecays();
      hastauolapp=true;
    }
    if ( curSet == "Photos" || curSet == "Photos2155Legacy"){
      if ( !fPhotosInterface ){
	fPhotosInterface = (PhotosInterfaceBase*)(PhotosFactory::get()->create("Photos2155Legacy", pset.getUntrackedParameter< ParameterSet >(curSet)));
	fPhotosInterface->SetDecayRandomEngine(decayRandomEngine);
      }
    }
    if (curSet == "Photos2155"){
      if ( !fPhotosInterface ){
	fPhotosInterface = (PhotosInterfaceBase*)(PhotosFactory::get()->create("Photos2155", pset.getUntrackedParameter< ParameterSet>(curSet)));
	fPhotosInterface->SetDecayRandomEngine(decayRandomEngine);
      }
    }
  }
}

ExternalDecayDriver::~ExternalDecayDriver(){
  if ( fEvtGenInterface ) delete fEvtGenInterface;
  if ( fTauolaInterface ) delete fTauolaInterface;
  if ( fPhotosInterface ) delete fPhotosInterface;
}

HepMC::GenEvent* ExternalDecayDriver::decay( HepMC::GenEvent* evt ){
  
  if ( !fIsInitialized ) return evt;
  
  if ( fEvtGenInterface ){  
    evt = fEvtGenInterface->decay( evt ); 
    if ( !evt ) return 0;
  }
  
  if ( fTauolaInterface ){
    evt = fTauolaInterface->decay( evt ); 
    if ( !evt ) return 0;
  }
  
  if ( fPhotosInterface ){
    evt = fPhotosInterface->apply( evt );
    if ( !evt ) return 0;
  }

  return evt;
}

void ExternalDecayDriver::init( const edm::EventSetup& es ){

   if ( fIsInitialized ) return;
   
   if ( fTauolaInterface ) {
     fTauolaInterface->init( es );
     for ( std::vector<int>::const_iterator i=fTauolaInterface->operatesOnParticles().begin();
	   i!=fTauolaInterface->operatesOnParticles().end(); i++ ) 
       fPDGs.push_back( *i );
   }
   if ( fEvtGenInterface ){
     fEvtGenInterface->init();
     for ( std::vector<int>::const_iterator i=fEvtGenInterface->operatesOnParticles().begin();
	   i!=fEvtGenInterface->operatesOnParticles().end(); i++ )
       fPDGs.push_back( *i );
   }
   
   
   if( fPhotosInterface){
     fPhotosInterface->init();
     if(hastauolapp){
       if ( fPhotosInterface ){
	 for ( unsigned int iss=0; iss<fPhotosInterface->specialSettings().size(); iss++ ){
	   fSpecialSettings.push_back( fPhotosInterface->specialSettings()[iss]);
	 }
       }
     }
   }
   // now put in hack for TauolaFortran settings
   if(!hastauolapp){
     if ( fPhotosInterface ){
       fSpecialSettings.push_back( "QED-brem-off:all" );
     }
     if ( fTauolaInterface ){
       // override !
       fSpecialSettings.clear();
       fSpecialSettings.push_back( "QED-brem-off:15" );
     }
   }
   fIsInitialized = true;
   
   return;
}

void ExternalDecayDriver::statistics() const
{
   if ( fTauolaInterface ) fTauolaInterface->statistics();
   // similar for EvtGen and/or Photos, if needs be
   return;
}
