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
	 void setSLHAFromHeader( const std::vector<std::string> &lines );
	 
	 void openSLHA( const char* );
	 void closeSLHA();
	      
     private:
        std::vector<std::string> fParamGeneral;
        std::vector<std::string> fParamCSA;
        std::vector<std::string> fParamSLHA; 
	int  fUnitSLHA;
   };

}

#endif
