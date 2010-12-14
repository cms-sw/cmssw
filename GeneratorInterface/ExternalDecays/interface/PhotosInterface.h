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

namespace gen {

   class PhotosInterface
   {
      public:
      
      // ctor & dtor
      PhotosInterface(); 
      PhotosInterface( const edm::ParameterSet& );
      ~PhotosInterface() {}

      void init();
      const std::vector<std::string>& specialSettings() { return fSpecialSettings; }
      HepMC::GenEvent* apply( HepMC::GenEvent* );
      void configureOnlyFor( int );
      
      private: 
            
      int                      fOnlyPDG;
      std::vector<std::string> fSpecialSettings;      
      bool                     fIsInitialized;
      
      
      struct Scaling {
         HepMC::ThreeVector weights;
	 int                flag;
	 Scaling( HepMC::ThreeVector vec, int flg ) 
	    : weights(HepMC::ThreeVector(1.,1.,1)), flag(1) { weights=vec; flag=flg; }
      };
       
   };

}

#endif
