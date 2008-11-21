   
// -*- C++ -*-

#include "Pythia6Hadronizer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HepMC/GenEvent.h"
#include "HepMC/PdfInfo.h"
#include "HepMC/PythiaWrapper6_2.h"
#include "HepMC/HEPEVT_Wrapper.h"
#include "HepMC/IO_HEPEVT.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

HepMC::IO_HEPEVT conv;

namespace gen
{

extern "C" {
   void pygive_(const char *line, int length);

   static bool call_pygive(const std::string &line)
   {
      int numWarn = pydat1.mstu[26];	// # warnings
      int numErr = pydat1.mstu[22];	// # errors

      pygive_(line.c_str(), line.length());

      return pydat1.mstu[26] == numWarn &&
             pydat1.mstu[22] == numErr;
   }

//   void pyupev_() ;   
//  void pyexec_();

} // extern "C"


Pythia6Hadronizer::Pythia6Hadronizer(edm::ParameterSet const& ps) 
   : HadronizerFtn(ps), 
     fPythiaListVerbosity(ps.getUntrackedParameter<int>("pythiaPylistVerbosity", 0))
{ 

   // Set PYTHIA parameters in a single ParameterSet
   edm::ParameterSet pythia_params = 
      ps.getParameter<edm::ParameterSet>("PythiaParameters") ;
      
      std::cout << " PythiaParameters found " << std::endl;

   // read and sort Pythia6 cards
   //
   std::vector<std::string> setNames =
      pythia_params.getParameter<std::vector<std::string> >("parameterSets");
      
      std::cout << " parameterSets found " << std::endl;

   std::vector<std::string>	paramLines;

   for(std::vector<std::string>::const_iterator iter = setNames.begin();
	                                        iter != setNames.end(); ++iter) 
   {
		std::vector<std::string> lines =
			pythia_params.getParameter< std::vector<std::string> >(*iter);

		for(std::vector<std::string>::const_iterator line = lines.begin();
		                                             line != lines.end(); ++line ) 
		{
			if (line->substr(0, 7) == "MRPY(1)")
				throw cms::Exception("PythiaError")
					<< "Attempted to set random number"
					   " using Pythia command 'MRPY(1)'."
					   " Please use the"
					   " RandomNumberGeneratorService."
					<< std::endl;

			paramLines.push_back(*line);
		}
   }

   edm::Service<edm::RandomNumberGenerator> rng;
   std::ostringstream ss;
   ss << "MRPY(1)=" << rng->mySeed();
   paramLines.push_back(ss.str());
	
   // now pass config cards to
   for(std::vector<std::string>::const_iterator iter = paramLines.begin();
	                                        iter != paramLines.end(); ++iter)
   {
		if (!call_pygive(*iter))
			throw cms::Exception("PythiaError")
				<< "Pythia did not accept \""
				<< *iter << "\"." << std::endl;
   }

}

bool Pythia6Hadronizer::doEvent()
{
   
   // generate event with Pythia6
   //
   call_pyevnt();
   // pyexec_();
   
   // convert to HEPEVT
   //
   call_pyhepc(1);
   
   // convert to HepMC
   //
   // HepMC::GenEvent* evt = conv.read_next_event();
   fGenEvent = conv.read_next_event();
   
   if (fMaxEventsToPrint > 0) 
   {
      fMaxEventsToPrint--;
      if (fPythiaListVerbosity) call_pylist(fPythiaListVerbosity);
      if (fHepMCVerbosity) 
      {
         std::cout << "Event process = " << pypars.msti[0] << std::endl 
	      << "----------------------" << std::endl;
         fGenEvent->print();
      }
   }


   return true;
}

bool Pythia6Hadronizer::generatePartonsAndHadronize()
{
   
   doEvent() ;
   
   // fill up HEPRUP & HEPEUP
   //
   
   return true;
}

bool Pythia6Hadronizer::hadronize()
{
   // shuffle around with FortranCallBacks...
   //
   assert(!fortranCallback.fInstance);
   fortranCallback.fInstance = this;
   
   doEvent();
   
   // shuffle around with FortranCallBacks...
   //
   fortranCallback.fInstance = 0;
   
   return true;
}

bool Pythia6Hadronizer::decay()
{
   return true;
}

bool Pythia6Hadronizer::initializeForExternalPartons()
{

   assert(!fortranCallback.fInstance);
   fortranCallback.fInstance = this;
     
   call_pygive("MSEL=0");
/*
   call_pygive(std::string("MSTP(143)=") +
	      (wantsShoweredEvent() ? "1" : "0"));
*/
   call_pyinit("USER", "", "", 0.0);
   
   fortranCallback.fInstance = 0;
   
   return true;
}

bool Pythia6Hadronizer::initializeForInternalPartons()
{
    
   // call_pygive("MSTP(111)=0");
   call_pyinit("CMS", "p", "p", fCOMEnergy);
   return true;
}

bool Pythia6Hadronizer::declareStableParticles()
{
   return true;
}

const char* Pythia6Hadronizer::classname() const
{
   return "gen::Pythia6Hadronizer";
}

}
