#ifndef gen_Pythia6PtGun_h
#define gen_Pythia6PtGun_h

#include "Pythia6Gun.h"

namespace gen {

   class Pythia6PtGun : public Pythia6Gun
   {
   
      public:
      
      Pythia6PtGun( const edm::ParameterSet& );
      virtual ~Pythia6PtGun();
      // void produce( edm::Event&, const edm::EventSetup& ) ;
      
      protected:
         void generateEvent() ;
      
      private:
      
         double  fMinPt ;
         double  fMaxPt ;
	 bool    fAddAntiParticle;
   
   };


}

#endif
