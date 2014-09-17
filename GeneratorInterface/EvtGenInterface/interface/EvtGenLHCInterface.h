#ifndef gen_EvtGenLHCInterface_h
#define gen_EvtGenLHCInterface_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

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
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenInterfaceBase.h"

class myEvtRandomEngine;


namespace HepMC {
  class GenParticle;
  class GenEvent;
}

namespace gen {
   class Pythia6Service;

   class EvtGenLHCInterface : public EvtGenInterfaceBase {
     public:
     
      // ctor & dtor
      EvtGenLHCInterface( const edm::ParameterSet& );
      ~EvtGenLHCInterface();

      void init();
      const std::vector<int>& operatesOnParticles() { return m_PDGs; }      
      HepMC::GenEvent* decay( HepMC::GenEvent* );
      void addToHepMC(HepMC::GenParticle* partHep, EvtId idEvt, HepMC::GenEvent* theEvent, bool del_daug);
      void go_through_daughters(EvtParticle* part);
      void update_candlist( int theIndex, HepMC::GenParticle *thePart );
      void setRandomEngine(CLHEP::HepRandomEngine* v);
      static double flat();

      // from Pythia
      // void call_pygive(const std::string& iParm );

      private:
      
      Pythia6Service* m_Py6Service;
            
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

      myEvtRandomEngine* the_engine;

      bool useDefault;
      std::string decay_table_s;
      std::string pdt_s;
      std::string user_decay_s;
      std::vector<std::string> forced_names;

      static CLHEP::HepRandomEngine* fRandomEngine;
   };
}
#endif
