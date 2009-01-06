   
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

#include "GeneratorInterface/Core/interface/FortranCallback.h"

#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "GeneratorInterface/Core/interface/RNDMEngineAccess.h"

#include "GeneratorInterface/Pythia6Interface/interface/PYR.h"


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

   void upinit_() { FortranCallback::getInstance()->fillHeader(); return; }
   void upevnt_() { FortranCallback::getInstance()->fillEvent();  return ; }

} // extern "C"


Pythia6Hadronizer::Pythia6Hadronizer(edm::ParameterSet const& ps) 
   : fCOMEnergy(ps.getParameter<double>("comEnergy")),
     fGenEvent(0),
     fEventCounter(0),
     fRandomEngine(getEngineReference()),     
     fHepMCVerbosity(ps.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
     fMaxEventsToPrint(ps.getUntrackedParameter<int>("maxEventsToPrint", 0)),
     fPythiaListVerbosity(ps.getUntrackedParameter<int>("pythiaPylistVerbosity", 0))
{ 

   // Set PYTHIA parameters in a single ParameterSet
   //
   edm::ParameterSet pythia_params = 
      ps.getParameter<edm::ParameterSet>("PythiaParameters") ;
      
   // read and sort Pythia6 cards
   //
   std::vector<std::string> setNames =
      pythia_params.getParameter<std::vector<std::string> >("parameterSets");
      
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

/* old stuff
   edm::Service<edm::RandomNumberGenerator> rng;
   std::ostringstream ss;
   ss << "MRPY(1)=" << rng->mySeed();
   paramLines.push_back(ss.str());
*/
   // Initialize the random engine unconditionally
   //
   randomEngine = &fRandomEngine;
   fRandomGenerator = new CLHEP::RandFlat(fRandomEngine) ;

   // first of all, silence Pythia6 banner printout
   if (!call_pygive("MSTU(12)=12345")) 
   {
      throw edm::Exception(edm::errors::Configuration,"PythiaError") 
          <<" Pythia did not accept MSTU(12)=12345";
   }
   
   // now pass config cards 
   //
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
   
   // convert to HEPEVT
   //
   call_pyhepc(1);
   
   // convert to HepMC
   //
   fGenEvent = conv.read_next_event();
   
   fGenEvent->set_signal_process_id(pypars.msti[0]);
   fGenEvent->set_event_scale(pypars.pari[16]);
   // evt->set_event_number(numberEventsInRun() - remainingEvents() - 1);
   fGenEvent->set_event_number( fEventCounter );
   

   // get pdf info directly from Pythia6 and set it up into HepMC::GenEvent
   //
   int id1 = pyint1.mint[14];
   int id2 = pyint1.mint[15];
   if ( id1 == 21 ) id1 = 0;
   if ( id2 == 21 ) id2 = 0; 
   double x1 = pyint1.vint[40];
   double x2 = pyint1.vint[41];  
   double Q  = pyint1.vint[50];
   double pdf1 = pyint1.vint[38];
   pdf1 /= x1 ;
   double pdf2 = pyint1.vint[39];
   pdf2 /= x2 ;
   fGenEvent->set_pdf_info( HepMC::PdfInfo(id1,id2,x1,x2,Q,pdf1,pdf2) ) ;
    
   fGenEvent->weights().push_back( pyint1.vint[96] );

   // service printouts, if requested
   //
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

   fEventCounter++;
   
   return true;
}

bool Pythia6Hadronizer::generatePartonsAndHadronize()
{
   
   doEvent() ;
      
   return true;
}

bool Pythia6Hadronizer::hadronize()
{
   
   doEvent();
      
   return true;
}

bool Pythia6Hadronizer::decay()
{
   return true;
}

bool Pythia6Hadronizer::initializeForExternalPartons()
{
     
   call_pygive("MSEL=0");
/*
   call_pygive(std::string("MSTP(143)=") +
	      (wantsShoweredEvent() ? "1" : "0"));
*/
   call_pyinit("USER", "", "", 0.0);
      
   return true;
}

bool Pythia6Hadronizer::initializeForInternalPartons()
{
    
   call_pyinit("CMS", "p", "p", fCOMEnergy);
   return true;
}

bool Pythia6Hadronizer::declareStableParticles()
{
   return true;
}

void Pythia6Hadronizer::statistics()
{

  double cs = pypars.pari[0]; // cross section in mb
  fGenInfoProduct.set_cross_section(cs);
  // fGenInfoProduct.set_external_cross_section(extCrossSect);
  // fGenInfoProduct.set_filter_efficiency(extFilterEff);

  call_pystat(1);
  
  return;

}

const char* Pythia6Hadronizer::classname() const
{
   return "gen::Pythia6Hadronizer";
}

void Pythia6Hadronizer::setLHERunInfoProd( LHERunInfoProduct* lherp )
{

   FortranCallback::getInstance()->setLHERunProd(lherp);

   return;

}

void Pythia6Hadronizer::setLHEEventProd( LHEEventProduct* lheep )
{

   FortranCallback::getInstance()->setLHEEventProd(lheep);

   return;

}

}
