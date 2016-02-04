#ifndef gen_Pythia6EGun_h
#define gen_Pythia6EGun_h

#include "Pythia6ParticleGun.h"

namespace gen {

   class Pythia6EGun : public Pythia6ParticleGun
   {
   
      public:
      
        Pythia6EGun( const edm::ParameterSet& );
        virtual ~Pythia6EGun();
        // void produce( edm::Event&, const edm::EventSetup& ) ;
      
      protected:
         void generateEvent();
      
      private:
      
         double  fMinEta;
	 double  fMaxEta;
	 double  fMinE ;
         double  fMaxE ;
	 bool    fAddAntiParticle;
   
   };


}

#endif
