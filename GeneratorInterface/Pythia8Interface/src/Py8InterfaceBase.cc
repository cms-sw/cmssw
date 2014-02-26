//#include <iostream>
//#include <sstream>
//#include <string>
//#include <memory>
//#include <stdint.h>

#include "GeneratorInterface/Pythia8Interface/interface/Py8InterfaceBase.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace Pythia8;

namespace gen {

Py8InterfaceBase::Py8InterfaceBase( edm::ParameterSet const& ps )
{
  fMasterGen.reset(new Pythia);
  fDecayer.reset(new Pythia);

  fMasterGen->readString("Next:numberShowEvent = 0");
  fDecayer->readString("Next:numberShowEvent = 0");

  fMasterGen->setRndmEnginePtr( &p8RndmEngine_ );
  fDecayer->setRndmEnginePtr( &p8RndmEngine_ );
  
  fParameters = ps.getParameter<edm::ParameterSet>("PythiaParameters");
  
  pythiaPylistVerbosity = ps.getUntrackedParameter<int>("pythiaPylistVerbosity", 0);
  pythiaHepMCVerbosity  = ps.getUntrackedParameter<bool>("pythiaHepMCVerbosity", false);
  maxEventsToPrint      = ps.getUntrackedParameter<int>("maxEventsToPrint", 0);

}

bool Py8InterfaceBase::readSettings( int ) 
{

   for ( ParameterCollector::const_iterator line = fParameters.begin();
         line != fParameters.end(); ++line ) 
   {
      if (line->find("Random:") != std::string::npos)
         throw cms::Exception("PythiaError") << "Attempted to set random number "
         "using Pythia commands. Please use " "the RandomNumberGeneratorService."
         << std::endl;

      if (!fMasterGen->readString(*line)) throw cms::Exception("PythiaError")
			              << "Pythia 8 did not accept \""
				      << *line << "\"." << std::endl;
   }

   return true;

}

bool Py8InterfaceBase::declareStableParticles( const std::vector<int>& pdgIds )
{
  
  for ( size_t i=0; i<pdgIds.size(); i++ )
  {
    // FIXME: need to double check if PID's are the same in Py6 & Py8,
    //        because the HepPDT translation tool is actually for **Py6** 
    // 
    // well, actually it looks like Py8 operates in PDT id's rather than Py6's
    //
    // int PyID = HepPID::translatePDTtoPythia( pdgIds[i] ); 
    int PyID = pdgIds[i]; 
    std::ostringstream pyCard ;
    pyCard << PyID <<":mayDecay=false";
    fMasterGen->readString( pyCard.str() );
    // alternative:
    // set the 2nd input argument warn=false 
    // - this way Py8 will NOT print warnings about unknown particle code(s)
    // fMasterPtr->readString( pyCard.str(), false )
  }
        
   return true;

}

bool Py8InterfaceBase:: declareSpecialSettings( const std::vector<std::string>& settings )
{

   for ( unsigned int iss=0; iss<settings.size(); iss++ )
   {
      if ( settings[iss].find("QED-brem-off") == std::string::npos ) continue;
      fMasterGen->readString( "TimeShower:QEDshowerByL=off" );
   }

   return true;
}


void Py8InterfaceBase::statistics()
{
  
   fMasterGen->statistics();
   return;
   
}

}
