#include "GeneratorInterface/Pythia8Interface/interface/Py8InterfaceBase.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// EvtGen plugin
//
//#include "Pythia8Plugins/EvtGen.h"

using namespace Pythia8;

namespace gen {

Py8InterfaceBase::Py8InterfaceBase( edm::ParameterSet const& ps ) :
useEvtGen(false), evtgenDecays(0)
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
  pythiaHepMCVerbosityParticles = ps.getUntrackedParameter<bool>("pythiaHepMCVerbosityParticles", false);
  maxEventsToPrint      = ps.getUntrackedParameter<int>("maxEventsToPrint", 0);

  if(pythiaHepMCVerbosityParticles)
    ascii_io = new HepMC::IO_AsciiParticles("cout", std::ios::out);

  if ( ps.exists("useEvtGenPlugin") ) {

    useEvtGen = true;

    string evtgenpath(getenv("EVTGENDATA"));
    evtgenDecFile = evtgenpath + string("/DECAY_2010.DEC");
    evtgenPdlFile = evtgenpath + string("/evt.pdl");

    if ( ps.exists( "evtgenDecFile" ) )
      evtgenDecFile = ps.getParameter<string>("evtgenDecFile");

    if ( ps.exists( "evtgenPdlFile" ) )
      evtgenPdlFile = ps.getParameter<string>("evtgenPdlFile");

  }

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

      if (line->find("ParticleDecays:") != std::string::npos) {

        if (!fDecayer->readString(*line)) throw cms::Exception("PythiaError")
                                      << "Pythia 8 Decayer did not accept \""
                                      << *line << "\"." << std::endl;
      }

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
//    int PyID = HepPID::translatePDTtoPythia( pdgIds[i] ); 
    int PyID = pdgIds[i]; 
    std::ostringstream pyCard ;
    pyCard << PyID <<":mayDecay=false";

    if ( fMasterGen->particleData.isParticle( PyID ) ) {
       fMasterGen->readString( pyCard.str() );
    } else {

       edm::LogWarning("DataNotUnderstood") << "Pythia8 does not "
                                            << "recognize particle id = " 
                                            << PyID << std::endl;
    } 
    // alternative:
    // set the 2nd input argument warn=false 
    // - this way Py8 will NOT print warnings about unknown particle code(s)
    // fMasterPtr->readString( pyCard.str(), false )
  }
        
   return true;

}

bool Py8InterfaceBase:: declareSpecialSettings( const std::vector<std::string>& settings ){
   for ( unsigned int iss=0; iss<settings.size(); iss++ ){
     if ( settings[iss].find("QED-brem-off") != std::string::npos ){
       fMasterGen->readString( "TimeShower:QEDshowerByL=off" );
     }
     else{
       size_t fnd1 = settings[iss].find("Pythia8:");
       if ( fnd1 != std::string::npos ){
	 std::string value = settings[iss].substr (fnd1+8);
	 fDecayer->readString(value);
       }
     }
   }
   return true;
}


void Py8InterfaceBase::statistics()
{
  
   fMasterGen->stat();
   return;
   
}

}
