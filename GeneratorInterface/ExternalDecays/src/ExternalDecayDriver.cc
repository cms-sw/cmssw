
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/Core/interface/FortranInstance.h"

#include "GeneratorInterface/ExternalDecays/interface/EvtGenInterface.h"
#include "GeneratorInterface/ExternalDecays/interface/TauolaInterface.h"
#include "GeneratorInterface/ExternalDecays/interface/PhotosInterface.h"

#include "GeneratorInterface/ExternalDecays/interface/DecayRandomEngine.h"

#include "HepMC/GenEvent.h"

#include "FWCore/Concurrency/interface/SharedResourceNames.h"

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

    for (unsigned int ip=0; ip<extGenNames.size(); ++ip )
    {
      std::string curSet = extGenNames[ip];
      if ( curSet == "EvtGen" )
      {
         fEvtGenInterface = new gen::EvtGenInterface(pset.getUntrackedParameter< ParameterSet >(curSet));
         exSharedResources.emplace_back(edm::SharedResourceNames::kEvtGen);
         exSharedResources.emplace_back(edm::SharedResourceNames::kPythia6);
         exSharedResources.emplace_back(gen::FortranInstance::kFortranInstance);
      }
      else if ( curSet == "Tauola" )
      {
         // this is for old tauola27 (+pretauola)
	 //
	 // --> fTauolaInterface = new gen::TauolaInterface(pset.getUntrackedParameter< ParameterSet >(curSet));
	 //
	 // for tauola++, here it should be something like:
	 //
	 fTauolaInterface = TauolaInterface::getInstance();
	 fTauolaInterface->setPSet( pset.getUntrackedParameter< ParameterSet >(curSet) );
	 fPhotosInterface = new gen::PhotosInterface();
	 fPhotosInterface->configureOnlyFor( 15 );
	 fPhotosInterface->avoidTauLeptonicDecays();
         exSharedResources.emplace_back(edm::SharedResourceNames::kTauola);
      }
      else if ( curSet == "Photos" )
      {
         if ( !fPhotosInterface ) fPhotosInterface = new gen::PhotosInterface();
         exSharedResources.emplace_back(edm::SharedResourceNames::kPhotos);
      }
    }
}

ExternalDecayDriver::~ExternalDecayDriver()
{
   if ( fEvtGenInterface ) delete fEvtGenInterface;
   if ( fTauolaInterface ) delete fTauolaInterface;
   if ( fPhotosInterface ) delete fPhotosInterface;
}

HepMC::GenEvent* ExternalDecayDriver::decay( HepMC::GenEvent* evt )
{
   
   if ( !fIsInitialized ) return evt;
   
   if ( fEvtGenInterface )
   {  
      evt = fEvtGenInterface->decay( evt ); 
      if ( !evt ) return 0;
   }

   if ( fTauolaInterface ) 
   {
      evt = fTauolaInterface->decay( evt ); 
      if ( !evt ) return 0;
   }
   
   if ( fPhotosInterface )
   {
      evt = fPhotosInterface->apply( evt );
      if ( !evt ) return 0;
   }
         
   return evt;
}

void ExternalDecayDriver::init( const edm::EventSetup& es )
{

   if ( fIsInitialized ) return;
   
   if ( fTauolaInterface ) 
   {
      fTauolaInterface->init( es );
      for ( std::vector<int>::const_iterator i=fTauolaInterface->operatesOnParticles().begin();
            i!=fTauolaInterface->operatesOnParticles().end(); i++ ) 
               fPDGs.push_back( *i );
   }
   
   if ( fEvtGenInterface ) 
   {
      fEvtGenInterface->init();
      for ( std::vector<int>::const_iterator i=fEvtGenInterface->operatesOnParticles().begin();
            i!=fEvtGenInterface->operatesOnParticles().end(); i++ )
               fPDGs.push_back( *i );
   }
   

   if ( fPhotosInterface )
   {
      fPhotosInterface->init();
//   for tauola++ 
      if ( fPhotosInterface )
      {
         for ( unsigned int iss=0; iss<fPhotosInterface->specialSettings().size(); iss++ )
         {
            fSpecialSettings.push_back( fPhotosInterface->specialSettings()[iss] );
         }
      }
   }
   
// this is specific to old tauola27 only, because it calls up photos automatically
//
//
//   if ( fTauolaInterface )
//   {
//      // override !
//      fSpecialSettings.clear();
//      fSpecialSettings.push_back( "QED-brem-off:15" );
//   }
   
   fIsInitialized = true;
   
   return;
}

void ExternalDecayDriver::statistics() const
{
   if ( fTauolaInterface ) fTauolaInterface->statistics();
   // similar for EvtGen and/or Photos, if needs be
   return;
}

void ExternalDecayDriver::setRandomEngine(CLHEP::HepRandomEngine* v)
{
   decayRandomEngine = v;
   if ( fTauolaInterface ) {
      fTauolaInterface->setRandomEngine(v);
   }
   if ( fEvtGenInterface ) {
      fEvtGenInterface->setRandomEngine(v);
   }
}
