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

namespace CLHEP
{
class HepRandomEngine;
}

namespace gen {

/* for old tauola27 
   class TauolaInterface
   {
      public:
      
      // ctor & dtor
      TauolaInterface( const edm::ParameterSet& );
      ~TauolaInterface();
      
      void enablePolarization()  { fPolarization = 1; return; }
      void disablePolarization() { fPolarization = 0; return; }
      void init( const edm::EventSetup& );
      const std::vector<int>& operatesOnParticles() { return fPDGs; }
      HepMC::GenEvent* decay( HepMC::GenEvent* );
      void statistics() ;
      
      private: 
      
      //            
      std::vector<int> fPDGs;
      int              fPolarization;
      
      edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
      Pythia6Service*                          fPy6Service;
      bool                                     fIsInitialized;
      //CLHEP::HepRandomEngine*                  fRandomEngine;
      //CLHEP::RandFlat*                         fRandomGenerator;
       
   };
*/

/* this is the code for new Tauola++ */

   extern "C" {
      void ranmar_( float *rvec, int *lenv );
      void rmarin_( int*, int*, int* );
   }

   double TauolappInterface_RandGetter();

   class TauolaInterface
   {
      public:
      
      // ctor & dtor
      // TauolaInterface( const edm::ParameterSet& );
      static TauolaInterface* getInstance() ;
      ~TauolaInterface();
      
      void setPSet( const edm::ParameterSet& );
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
      TauolaInterface();
      
      // member function(s)
      float flat();
      void decodeMDTAU( int );
      void selectDecayByMDTAU();
      int selectLeptonic();
      int selectHadronic();
      
      
      //
      CLHEP::HepRandomEngine*                  fRandomEngine;            
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
      
      static TauolaInterface*                  fInstance;
       
   };


/* */

}

#endif
