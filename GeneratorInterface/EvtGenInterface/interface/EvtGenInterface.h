#ifndef gen_EvtGenInterface_h
#define gen_EvtGenInterface_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenInterfaceBase.h"

namespace CLHEP {
  class HepRandomEngine;
  class RandFlat;
}

namespace HepMC {
  class GenParticle;
  class GenEvent;
}

class EvtGen;
class EvtId;
class EvtParticle;

namespace gen {

   class Pythia6Service;

   class EvtGenInterface : public EvtGenInterfaceBase {
   public:
     EvtGenInterface( const edm::ParameterSet& );
     ~EvtGenInterface();

     void SetPhotosDecayRandomEngine(CLHEP::HepRandomEngine* decayRandomEngine);     
     void init();
     const std::vector<int>& operatesOnParticles() { return m_PDGs; }      
     HepMC::GenEvent* decay( HepMC::GenEvent* evt);

     void addToHepMC(HepMC::GenParticle* partHep, EvtId idEvt, HepMC::GenEvent* theEvent, bool del_daug);
     void go_through_daughters(EvtParticle* part);
     void update_candlist( int theIndex, HepMC::GenParticle *thePart );

   private:
     // from Pythia
     // void call_pygive(const std::string& iParm );
     
      Pythia6Service* m_Py6Service;
      CLHEP::RandFlat* m_flat;   
      EvtGen *m_EvtGen;
      std::vector<EvtId> forced_Evt;     // EvtId's of particles with forced decay
      std::vector<int> forced_Hep;       // HepId's of particles with forced decay
      int nforced;                       // number of particles with forced decay
      int ntotal, npartial, nevent;      // generic counters
      
      int nPythia;
      bool usePythia;
      // std::vector<std::string> pythia_params;  // Pythia stuff

      // Adding parameters for polarization of spin-1/2 particles
      std::vector<int> polarize_ids;
      std::vector<double> polarize_pol;
      std::map<int, float> polarizations;
      
      int nlist; 
      HepMC::GenParticle *listp[10]; 
      int index[10];                     // list of candidates to be forced  
       
   };
}

#endif
