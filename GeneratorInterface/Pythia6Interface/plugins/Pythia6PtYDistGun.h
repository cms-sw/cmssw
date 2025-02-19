#ifndef gen_Pythia6PtYDistGun_h
#define gen_Pythia6PtYDistGun_h

#include "Pythia6ParticleGun.h"
// #include "GeneratorInterface/Pythia6Interface/interface/PtYDistributor.h"


namespace gen {

class PtYDistributor;

   class Pythia6PtYDistGun : public Pythia6ParticleGun
   {
   
      public:
      
      Pythia6PtYDistGun( const edm::ParameterSet& );
      virtual ~Pythia6PtYDistGun();
      // void produce( edm::Event&, const edm::EventSetup& ) ;
      
      protected:
         void generateEvent() ;
      
      private:
      
	 PtYDistributor* fPtYGenerator; 
   
   };


}

#endif
