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

   class TauolaInterface
   {
      public:
      
      // ctor & dtor
      TauolaInterface( const edm::ParameterSet& );
      ~TauolaInterface();
      
      void init( const edm::EventSetup& );
      const std::vector<int>& operatesOnParticles() { return fPDGs; }
      HepMC::GenEvent* decay( const HepMC::GenEvent* );
      void statistics() ;
      
      private: 
      
      //            
      std::vector<int> fPDGs;
      int              fPolarization;
      
      edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
      //CLHEP::HepRandomEngine*             fRandomEngine ;
      //CLHEP::RandFlat*                    fRandomGenerator;
       
   };

}

#endif
