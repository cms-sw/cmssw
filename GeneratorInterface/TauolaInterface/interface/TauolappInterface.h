#ifndef gen_TauolaInterface_TauolappInterface_h
#define gen_TauolaInterface_TauolappInterface_h

#include "HepPDT/ParticleDataTable.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaInterfaceBase.h"

namespace HepMC 
{
class GenEvent;
}

namespace CLHEP
{
class HepRandomEngine;
}

namespace gen {
  extern "C" {
    void ranmar_( float *rvec, int *lenv );
    void rmarin_( int*, int*, int* );
  }

   class TauolappInterface : public TauolaInterfaceBase {
      public:
      
      // ctor & dtor
     TauolappInterface( const edm::ParameterSet& );
     ~TauolappInterface();
     
      void enablePolarization()  { fPolarization = true; return; }
      void disablePolarization() { fPolarization = false; return; }
      void init( const edm::EventSetup& );
      const std::vector<int>& operatesOnParticles() { return fPDGs; }
      HepMC::GenEvent* decay( HepMC::GenEvent* );
      void statistics() ;

      void setRandomEngine(CLHEP::HepRandomEngine* v) { fRandomEngine = v; }
      static double flat();
      
      private: 
      // member function(s)
      void decodeMDTAU( int );
      void selectDecayByMDTAU();
      int selectLeptonic();
      int selectHadronic();
      
      //
      static CLHEP::HepRandomEngine*           fRandomEngine;            
      std::vector<int>                         fPDGs;
      bool                                     fPolarization;      
      edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
      edm::ParameterSet*                       fPSet;
      bool                                     fIsInitialized;
      
      int                                      fMDTAU;
      bool                                     fSelectDecayByEvent;
      std::vector<int>                         fLeptonModes;
      std::vector<int>                         fHadronModes;
      std::vector<double>                      fScaledLeptonBrRatios;
      std::vector<double>                      fScaledHadronBrRatios;
      
   };

}

#endif
