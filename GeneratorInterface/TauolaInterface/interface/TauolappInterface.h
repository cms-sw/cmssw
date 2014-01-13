#ifndef gen_TauolaInterface_TauolappInterface_h
#define gen_TauolaInterface_TauolappInterface_h

// #include "HepPDT/defs.h"
// #include "HepPDT/TableBuilder.hh"
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

   double TauolappInterface_RandGetter();

   class TauolappInterface : public TauolaInterfaceBase {
      public:
      
      // ctor & dtor
     TauolappInterface( );
     TauolappInterface( const edm::ParameterSet&);
      static TauolappInterface* getInstance() ;
      ~TauolappInterface();
      
      void setPSet( const edm::ParameterSet& );
      void Setup();
      void enablePolarization()  { fPolarization = true; return; }
      void disablePolarization() { fPolarization = false; return; }
      void init( const edm::EventSetup& );
      const std::vector<int>& operatesOnParticles() { return fPDGs; }
      HepMC::GenEvent* decay( HepMC::GenEvent* );
      void statistics() ;
      
      private: 
      
      friend void gen::ranmar_( float *rvec, int *lenv );
      friend double gen::TauolappInterface_RandGetter();      
      // ctor
      //TauolappInterface();
      
      // member function(s)
      float flat();
      void decodeMDTAU( int );
      void selectDecayByMDTAU();
      int selectLeptonic();
      int selectHadronic();
      
      
      //
      CLHEP::HepRandomEngine*                  fRandomEngine;            
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
      
      static TauolappInterface*                  fInstance;
       
   };


/* */

}

#endif
