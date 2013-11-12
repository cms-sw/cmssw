#ifndef gen_EvtGenInterface_EvtGenInterfaceBase_h
#define gen_EvtGenInterface_EvtGenInterfaceBase_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "HepMC/GenEvent.h"
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {
   class EvtGenInterfaceBase {
   public:
     EvtGenInterfaceBase(){ };
     virtual ~EvtGenInterfaceBase(){ };

     virtual void SetPhotosDecayRandomEngine(CLHEP::HepRandomEngine* decayRandomEngine){};
     virtual void init(){};
     virtual const std::vector<int>& operatesOnParticles(){return m_PDGs;}
     virtual HepMC::GenEvent* decay( HepMC::GenEvent* evt){return evt;}
     
   protected: 
     std::vector<int> m_PDGs;

   };
}

#endif
