#ifndef gen_TauolaInterface_TauolaInterfaceBase_h
#define gen_TauolaInterface_TauolaInterfaceBase_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "HepMC/GenEvent.h"
#include <vector>
#include "CLHEP/Random/RandomEngine.h"

namespace gen {
   class TauolaInterfaceBase {
   public:
     TauolaInterfaceBase(){};
     TauolaInterfaceBase( const edm::ParameterSet&){};
     virtual ~TauolaInterfaceBase(){};
     
     virtual void SetDecayRandomEngine(CLHEP::HepRandomEngine* decayRandomEngine){};
     virtual void enablePolarization(){};
     virtual void disablePolarization(){};
     virtual void init( const edm::EventSetup& ){};
     virtual const std::vector<int>& operatesOnParticles() { return fPDGs; }
     virtual HepMC::GenEvent* decay( HepMC::GenEvent* evt){return evt;}
     virtual void statistics(){};
     
   protected: 
     std::vector<int> fPDGs;
             
   };
}

#endif
