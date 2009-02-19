
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/ExternalDecays/interface/EvtGenInterface.h"
#include "GeneratorInterface/ExternalDecays/interface/TauolaInterface.h"

#include "HepMC/GenEvent.h"

using namespace gen;
using namespace edm;

ExternalDecayDriver::ExternalDecayDriver( const ParameterSet& pset )
   : fTauolaInterface(0),
     fEvtGenInterface(0)
{
    
    std::vector<std::string> extGenNames =
       pset.getParameter< std::vector<std::string> >("parameterSets");
    
    for (unsigned int ip=0; ip<extGenNames.size(); ++ip )
    {
      std::string curSet = extGenNames[ip];
      if ( curSet == "EvtGen" )
      {
         fEvtGenInterface = new gen::EvtGenInterface(pset.getUntrackedParameter< ParameterSet >(curSet));
      }
      else if ( curSet == "Tauola" )
      {
         fTauolaInterface = new gen::TauolaInterface(pset.getUntrackedParameter< ParameterSet >(curSet));
      }
    }

}

ExternalDecayDriver::~ExternalDecayDriver()
{
   if ( fEvtGenInterface ) delete fEvtGenInterface;
   if ( fTauolaInterface ) delete fTauolaInterface;
}

HepMC::GenEvent* ExternalDecayDriver::decay( HepMC::GenEvent* evt )
{
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
         
   return evt;
}

void ExternalDecayDriver::init()
{

   if ( fTauolaInterface ) 
   {
      fTauolaInterface->init();
      for ( std::vector<int>::const_iterator i=fTauolaInterface->operatesOnParticles().begin();
            i != fTauolaInterface->operatesOnParticles().end(); i++ ) 
               fPDGs.push_back( *i );
   }
   // if ( fEvtGenInterface ) 
   //{
   //   fEvtGenInterface->init();
   //   fPDGs += fEvtGenInterface->operatesOnParticles();
   //}
   
   return;
}

void ExternalDecayDriver::statistics() const
{
   if ( fTauolaInterface ) fTauolaInterface->statistics();
   // similar for EvtGen, if needs be
   return;
}
