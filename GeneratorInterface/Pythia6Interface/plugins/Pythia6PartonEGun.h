#ifndef gen_Pythia6PtGun_h
#define gen_Pythia6PtGun_h

#include "Pythia6PartonGun.h"

namespace gen {

   class Pythia6PartonEGun : public Pythia6PartonGun
   {
   
      public:
      
      Pythia6PartonEGun( const edm::ParameterSet& );
      virtual ~Pythia6PartonEGun();
      
      protected:

         void generateEvent() ;
      
      private:
      
         double  fMinEta;
	 double  fMaxEta;
	 double  fMinE ;
         double  fMaxE ;
   
   };


}

#endif
