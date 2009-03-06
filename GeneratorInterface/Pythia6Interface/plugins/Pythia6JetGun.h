#ifndef gen_Pythia6JetGun_h
#define gen_Pythia6JetGun_h

#include "Pythia6Gun.h"

namespace gen {

   class Pythia6JetGun : public Pythia6Gun
   {
   
      public:
      
      Pythia6JetGun( const edm::ParameterSet& );
      virtual ~Pythia6JetGun();
      // void produce( edm::Event&, const edm::EventSetup& ) ;
      
      protected:
         void generateEvent() ;
      
      private:
      
	 double  fMinE ;
	 double  fMaxE ;
         double  fMinP ;
         double  fMaxP ;
   
   };

}

#endif
