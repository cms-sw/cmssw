#ifndef gen_ExternalDecays_PhotosInterface_h
#define gen_ExternalDecays_PhotosInterface_h

// #include "HepPDT/ParticleDataTable.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "HepMC/SimpleVector.h"

namespace HepMC 
{
class GenEvent;
}

#include "CLHEP/Random/RandomEngine.h"
extern CLHEP::HepRandomEngine* photosRandomEngine;

namespace gen {

   class PhotosInterface
   {
      public:
      
      // ctor & dtor
      PhotosInterface() : fIsInitialized(false) {}
      PhotosInterface( const edm::ParameterSet& ) : fIsInitialized(false){}
      ~PhotosInterface() {}

      void init();
      const std::vector<int>& operatesOnParticles() { return fPDGs; }
      HepMC::GenEvent* apply( HepMC::GenEvent* );
      
      private: 
      
      // do I need this ???            
      std::vector<int> fPDGs;
      
      // Pythia6Service*                          fPy6Service;
      bool                                     fIsInitialized;
      
      struct Scaling {
         HepMC::ThreeVector weights;
	 int                flag;
	 Scaling( HepMC::ThreeVector vec, int flg ) 
	    : weights(HepMC::ThreeVector(1.,1.,1)), flag(1) { weights=vec; flag=flg; }
      };
       
   };

}

#endif
