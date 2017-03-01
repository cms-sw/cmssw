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
     virtual const std::vector<std::string>& specialSettings() { return fSpecialSettings; }
     virtual HepMC::GenEvent* decay( HepMC::GenEvent* evt){return evt;}
     virtual void setRandomEngine(CLHEP::HepRandomEngine* v)=0;

   protected: 
     std::vector<int> m_PDGs;
     std::vector<std::string> fSpecialSettings;

   };
}

#endif
