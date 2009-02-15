#ifndef gen_Pythia6Service_h
#define gen_Pythia6Service_h

#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/Exception.h"

// #include "HepMC/PythiaWrapper6_2.h"

namespace gen
{

   class Pythia6Service
   {
      public:         
         // ctor & dtor
	 Pythia6Service( edm::ParameterSet const& );
	 ~Pythia6Service() ; 
     
         void setGeneralParams();
         void setCSAParams();
         void setSLHAParams();
     
     private:
        std::vector<std::string> paramGeneral;
        std::vector<std::string> paramCSA;
        std::vector<std::string> paramSLHA;  
   };

}

#endif
