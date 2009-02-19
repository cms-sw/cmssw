#ifndef gen_Pythia6EGun_h
#define gen_Pythia6EGun_h

#include "Pythia6Gun.h"

namespace gen {

   class Pythia6EGun : public Pythia6Gun
   {
   
      public:
      
      Pythia6EGun( const edm::ParameterSet& );
      virtual ~Pythia6EGun();
      void produce( edm::Event&, const edm::EventSetup& ) ;
      
      private:
      
         double  fMinE ;
         double  fMaxE ;
	 bool    fAddAntiParticle;
   
   };


}

#endif
