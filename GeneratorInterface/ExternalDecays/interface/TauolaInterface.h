#ifndef gen_ExternalDecays_TauolaInterface_h
#define gen_ExternalDecays_TauolaInterface_h

// #include "HepPDT/defs.h"
// #include "HepPDT/TableBuilder.hh"
#include "HepPDT/ParticleDataTable.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace HepMC 
{
class GenEvent;
}

namespace gen {

   //class Pythia6Service;

   class TauolaInterface
   {
      public:
      
      // ctor & dtor
      TauolaInterface( const edm::ParameterSet& );
      ~TauolaInterface();
      
      void enablePolarization()  { fPolarization = 1; return; }
      void disablePolarization() { fPolarization = 0; return; }
      void init( const edm::EventSetup& );
      const std::vector<int>& operatesOnParticles() { return fPDGs; }
      HepMC::GenEvent* decay( HepMC::GenEvent* );
      void statistics() ;
      
      private: 
      
      //            
      std::vector<int> fPDGs;
      int              fPolarization;
      
      edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
      Pythia6Service*                          fPy6Service;
      bool                                     fIsInitialized;
      //CLHEP::HepRandomEngine*                  fRandomEngine;
      //CLHEP::RandFlat*                         fRandomGenerator;
       
   };

/* this is the code for new Tauola++ 

   class TauolaInterface
   {
      public:
      
      // ctor & dtor
      TauolaInterface( const edm::ParameterSet& );
      ~TauolaInterface();
      
      void enablePolarization()  { fPolarization = true; return; }
      void disablePolarization() { fPolarization = false; return; }
      void init( const edm::EventSetup& );
      const std::vector<int>& operatesOnParticles() { return fPDGs; }
      HepMC::GenEvent* decay( HepMC::GenEvent* );
      void statistics() ;
      
      private: 
      
      //            
      std::vector<int>                         fPDGs;
      bool                                     fPolarization;      
      edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
      bool                                     fIsInitialized;
       
   };

*/

}

#endif
