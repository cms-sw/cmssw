#ifndef gen_Py8InterfaceBase_h
#define gen_Py8InterfaceBase_h

#include <vector>
#include <string>

#include "GeneratorInterface/Core/interface/ParameterCollector.h"
#include "GeneratorInterface/Pythia8Interface/interface/P8RndmEngine.h"

#include "HepMC/IO_AsciiParticles.h"

#include <Pythia8/Pythia.h>
#include <Pythia8Plugins/HepMC2.h>

class EvtGenDecays;

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {

   class Py8InterfaceBase {

      public:
         
	 Py8InterfaceBase( edm::ParameterSet const& ps );
	 ~Py8InterfaceBase() {}
	 
         virtual bool generatePartonsAndHadronize() = 0;
         bool decay() { return true; } // NOT used - let's call it "design imperfection"
         bool readSettings( int ); // common func
         virtual bool initializeForInternalPartons() = 0;
         bool declareStableParticles( const std::vector<int>& ); // common func
         bool declareSpecialSettings( const std::vector<std::string>& ); // common func
         virtual void finalizeEvent() = 0; 
         virtual void statistics();
         virtual const char* classname() const = 0;

         void p8SetRandomEngine(CLHEP::HepRandomEngine* v) { p8RndmEngine_.setRandomEngine(v); }
         P8RndmEngine& randomEngine() { return p8RndmEngine_; }

      protected:
         
	 std::auto_ptr<Pythia8::Pythia> fMasterGen;
	 std::auto_ptr<Pythia8::Pythia> fDecayer;
	 HepMC::Pythia8ToHepMC          toHepMC;
	 ParameterCollector	        fParameters;
	 
	 unsigned int                   pythiaPylistVerbosity;
         bool                           pythiaHepMCVerbosity;
         bool                           pythiaHepMCVerbosityParticles;
	 unsigned int                   maxEventsToPrint;
         HepMC::IO_AsciiParticles*      ascii_io;

         // EvtGen plugin
         //
         bool useEvtGen;
         EvtGenDecays* evtgenDecays;
         std::string evtgenDecFile;
         std::string evtgenPdlFile;

      private:

         P8RndmEngine p8RndmEngine_;
   };
}
#endif
