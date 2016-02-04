#ifndef gen_EvtGenInterface_h
#define gen_EvtGenInterface_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CLHEP/Random/RandFlat.h"

#include "EvtGen/EvtGen.hh"     
#include "EvtGenBase/EvtId.hh"
#include "EvtGenBase/EvtPDL.hh"
#include "EvtGenBase/EvtDecayTable.hh"
#include "EvtGenBase/EvtSpinType.hh"
#include "EvtGenBase/EvtVector4R.hh"
#include "EvtGenBase/EvtParticle.hh"
#include "EvtGenBase/EvtScalarParticle.hh"
#include "EvtGenBase/EvtStringParticle.hh"
#include "EvtGenBase/EvtDiracParticle.hh"
#include "EvtGenBase/EvtVectorParticle.hh"
#include "EvtGenBase/EvtRaritaSchwingerParticle.hh"
#include "EvtGenBase/EvtTensorParticle.hh"
#include "EvtGenBase/EvtHighSpinParticle.hh"
#include "EvtGenBase/EvtStdHep.hh"
#include "EvtGenBase/EvtSecondary.hh"
#include "EvtGenModels/EvtPythia.hh"

namespace CLHEP {
  class HepRandomEngine;
  class RandFlat;
}

namespace HepMC {
  class GenParticle;
  class GenEvent;
}

namespace gen {

   class Pythia6Service;

   class EvtGenInterface
   {
      public:
      
      // ctor & dtor
      EvtGenInterface( const edm::ParameterSet& );
      ~EvtGenInterface();

      void init();
      const std::vector<int>& operatesOnParticles() { return m_PDGs; }      
      HepMC::GenEvent* decay( HepMC::GenEvent* );
      void addToHepMC(HepMC::GenParticle* partHep, EvtId idEvt, HepMC::GenEvent* theEvent, bool del_daug);
      void go_through_daughters(EvtParticle* part);
      void update_candlist( int theIndex, HepMC::GenParticle *thePart );
  
      // from Pythia
      // void call_pygive(const std::string& iParm );

      private:
      
      Pythia6Service* m_Py6Service;
      
      std::vector<int> m_PDGs;
      
      CLHEP::RandFlat* m_flat;   
      EvtGen *m_EvtGen;
      std::vector<EvtId> forced_Evt;     // EvtId's of particles with forced decay
      std::vector<int> forced_Hep;       // HepId's of particles with forced decay
      int nforced;                       // number of particles with forced decay
      int ntotal, npartial, nevent;      // generic counters
      
      int nPythia;
      bool usePythia;
      // std::vector<std::string> pythia_params;  // Pythia stuff
      
      int nlist; 
      HepMC::GenParticle *listp[10]; 
      int index[10];                     // list of candidates to be forced  
       
   };

}

#endif
