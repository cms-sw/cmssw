#ifndef gen_TauolaInterface_TauolaInterface_h
#define gen_TauolaInterface_TauolaInterface_h

#include "HepPDT/defs.h"
#include "HepPDT/TableBuilder.hh"
#include "HepPDT/ParticleDataTable.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CLHEP/Random/RandomEngine.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaInterfaceBase.h"

class Pythia6Service;

namespace HepMC 
{
class GenEvent;
}

namespace CLHEP
{
class HepRandomEngine;
}


namespace gen {

  class TauolaInterface : public TauolaInterfaceBase
   {
      public:
      
      // ctor & dtor
     TauolaInterface( const edm::ParameterSet&);
      ~TauolaInterface();

      void SetDecayRandomEngine(CLHEP::HepRandomEngine* decayRandomEngine);
      void enablePolarization()  { fPolarization = 1; return; }
      void disablePolarization() { fPolarization = 0; return; }
      void init( const edm::EventSetup& );
      const std::vector<int>& operatesOnParticles() { return fPDGs; }
      HepMC::GenEvent* decay( HepMC::GenEvent* );
      void statistics() ;
      
      private: 
      int              fPolarization;
      
      edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
      Pythia6Service*                          fPy6Service;
      bool                                     fIsInitialized;
      //CLHEP::HepRandomEngine*                  fRandomEngine;
      //CLHEP::RandFlat*                         fRandomGenerator;
       
   };
}

#endif
