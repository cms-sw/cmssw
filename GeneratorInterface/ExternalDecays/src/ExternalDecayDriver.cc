#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/Core/interface/FortranInstance.h"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenFactory.h"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenInterfaceBase.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaFactory.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaInterfaceBase.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosFactory.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosInterfaceBase.h"
#include "HepMC/GenEvent.h"
#include "FWCore/Concurrency/interface/SharedResourceNames.h"
// LHE Run
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

// LHE Event
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

using namespace gen;
using namespace edm;

CLHEP::HepRandomEngine* decayRandomEngine = nullptr;

ExternalDecayDriver::ExternalDecayDriver( const ParameterSet& pset )
   : fIsInitialized(false),
     fTauolaInterface(0),
     fEvtGenInterface(0),
     fPhotosInterface(0)
{
    std::vector<std::string> extGenNames =
    pset.getParameter< std::vector<std::string> >("parameterSets");
  
  for (unsigned int ip=0; ip<extGenNames.size(); ++ip ){
    std::string curSet = extGenNames[ip];
    if ( curSet == "EvtGen"){
      fEvtGenInterface = (EvtGenInterfaceBase*)(EvtGenFactory::get()->create("EvtGen", pset.getUntrackedParameter< ParameterSet >(curSet)));
      exSharedResources.emplace_back(edm::SharedResourceNames::kEvtGen);
      exSharedResources.emplace_back(edm::SharedResourceNames::kPythia6);
      exSharedResources.emplace_back(gen::FortranInstance::kFortranInstance);
    }
    else if( curSet == "EvtGen130"){
      fEvtGenInterface = (EvtGenInterfaceBase*)(EvtGenFactory::get()->create("EvtGen130", pset.getUntrackedParameter< ParameterSet >(curSet)));
      exSharedResources.emplace_back(edm::SharedResourceNames::kEvtGen);
      exSharedResources.emplace_back(edm::SharedResourceNames::kPythia8);
      exSharedResources.emplace_back(edm::SharedResourceNames::kTauola);
      exSharedResources.emplace_back(edm::SharedResourceNames::kPhotos);
      exSharedResources.emplace_back(gen::FortranInstance::kFortranInstance);
    }
    else if ( curSet == "Tauola" || curSet == "Tauolapp114" ){
      fTauolaInterface = (TauolaInterfaceBase*)(TauolaFactory::get()->create("Tauolapp114", pset.getUntrackedParameter< ParameterSet >(curSet)));
      fPhotosInterface = (PhotosInterfaceBase*)(PhotosFactory::get()->create("Photos2155", pset.getUntrackedParameter< ParameterSet >(curSet)));
      fPhotosInterface->configureOnlyFor( 15 );
      fPhotosInterface->avoidTauLeptonicDecays();
      exSharedResources.emplace_back(edm::SharedResourceNames::kTauola);
      exSharedResources.emplace_back(edm::SharedResourceNames::kPhotos);
    }
    else if ( curSet == "Photos" || curSet == "Photos2155" ){
      if ( !fPhotosInterface ){
	fPhotosInterface = (PhotosInterfaceBase*)(PhotosFactory::get()->create("Photos2155", pset.getUntrackedParameter< ParameterSet>(curSet)));
	exSharedResources.emplace_back(edm::SharedResourceNames::kPhotos);
      }
    }
    else if (curSet == "Photospp" || curSet == "Photospp356" ){
      if ( !fPhotosInterface ){
        fPhotosInterface = (PhotosInterfaceBase*)(PhotosFactory::get()->create("Photospp356", pset.getUntrackedParameter< ParameterSet>(curSet)));
        exSharedResources.emplace_back(edm::SharedResourceNames::kPhotos);
      }
    }
  }
}

ExternalDecayDriver::~ExternalDecayDriver()
{
   if ( fEvtGenInterface ) delete fEvtGenInterface;
   if ( fTauolaInterface ) delete fTauolaInterface;
   if ( fPhotosInterface ) delete fPhotosInterface;
}


HepMC::GenEvent* ExternalDecayDriver::decay(HepMC::GenEvent* evt, lhef::LHEEvent *lheEvent){
  if(fTauolaInterface) fTauolaInterface->SetLHE(lheEvent);
  return decay(evt);
}
HepMC::GenEvent* ExternalDecayDriver::decay( HepMC::GenEvent* evt )
{
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

void ExternalDecayDriver::init( const edm::EventSetup& es )
{

   if ( fIsInitialized ) return;
   
   if ( fTauolaInterface ){
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
       for ( unsigned int iss=0; iss<fEvtGenInterface->specialSettings().size(); iss++ ){
         fSpecialSettings.push_back( fEvtGenInterface->specialSettings()[iss] );
       }

   }
   
   
   if ( fPhotosInterface ){
     fPhotosInterface->init();
     //   for tauola++ 
     if ( fPhotosInterface ){
       for ( unsigned int iss=0; iss<fPhotosInterface->specialSettings().size(); iss++ ){
	 fSpecialSettings.push_back( fPhotosInterface->specialSettings()[iss] );
       }
     }
   }
   
   fIsInitialized = true;
   
   return;
}

void ExternalDecayDriver::statistics() const
{
   if ( fTauolaInterface ) fTauolaInterface->statistics();
   if ( fPhotosInterface ) fPhotosInterface->statistics();
   // similar for EvtGen if needed
   return;
}

void ExternalDecayDriver::setRandomEngine(CLHEP::HepRandomEngine* v)
{
   decayRandomEngine = v;
   if ( fTauolaInterface ) fTauolaInterface->setRandomEngine(v);
   if ( fEvtGenInterface ) fEvtGenInterface->setRandomEngine(v);
   if ( fPhotosInterface ) fPhotosInterface->setRandomEngine(v);
}
