#ifndef gen_Pythia6Service_h
#define gen_Pythia6Service_h

#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "GeneratorInterface/Core/interface/FortranInstance.h"

// #include "HepMC/PythiaWrapper6_2.h"

namespace CLHEP
{
   class HepRandomEngine;
}

namespace gen
{

   // the callbacks from Pythia which are passed on to the Pythia6Service
   extern "C" {
      double pyr_(int* idummy);
   }

   class Pythia6Service : public FortranInstance
   {
      public:         
         // ctor & dtor
         Pythia6Service();
	 Pythia6Service( edm::ParameterSet const& );
	 ~Pythia6Service() ; 
     
         void setGeneralParams();
         void setCSAParams();
         void setSLHAParams();
         void setPYUPDAParams(bool afterPyinit);
	 void setSLHAFromHeader( const std::vector<std::string> &lines );
	 
	 void openSLHA( const char* );
	 void closeSLHA();
	 void openPYUPDA( const char*, bool write_file );
	 void closePYUPDA();

         // initialise Pythia on first call from "dummy" instance
         virtual void enter();

     private:
        friend double gen::pyr_(int*);

        bool fInitialising;

        CLHEP::HepRandomEngine* fRandomEngine;

        std::vector<std::string> fParamGeneral;
        std::vector<std::string> fParamCSA;
        std::vector<std::string> fParamSLHA; 
        std::vector<std::string> fParamPYUPDA; 
	int  fUnitSLHA;
	int  fUnitPYUPDA;

        static Pythia6Service *fPythia6Owner;
   };

}

#endif
