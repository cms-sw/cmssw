#ifndef gen_Pythia6PtGun_h
#define gen_Pythia6PtGun_h

#include "Pythia6PartonGun.h"

namespace gen {

   class Pythia6PartonPtGun : public Pythia6PartonGun
   {
   
      public:
      
      Pythia6PartonPtGun( const edm::ParameterSet& );
      virtual ~Pythia6PartonPtGun();
      
      protected:

         void generateEvent() ;
      
      private:
      
         double  fMinEta;
	 double  fMaxEta;
	 double  fMinPt ;
         double  fMaxPt ;
   
   };


}

#endif
