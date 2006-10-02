#include "InitRootHandlers.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <TROOT.h>
#include <TSystem.h>
#include <TError.h>
#include <iostream>

namespace {
void rootErrorHandler( int level, bool die, const char* location, const char* message )
{
// Translate ROOT severity level to MessageLogger severity level

  edm::ELseverityLevel el_severity = edm::ELseverityLevel::ELsev_info ;

  if      (level >= kFatal)
    { el_severity = edm::ELseverityLevel::ELsev_fatal ;
    }
  else if (level >= kSysError)
    { el_severity = edm::ELseverityLevel::ELsev_severe ;
    }
  else if (level >= kError)
    { el_severity = edm::ELseverityLevel::ELsev_error ;
    }
  else if (level >= kWarning)
    { el_severity = edm::ELseverityLevel::ELsev_warning ;
    }

// Adapt C-strings to std::strings
// Arrange to report the error location as furnished by Root

  std::string el_location = "@SUB=?" ;
  if (location != 0) el_location = std::string("@SUB=")+std::string(location) ;

  std::string el_message  = "?" ;
  if (message != 0)  el_message  = message ;

// Try to create a meaningful id string using knowledge of ROOT error messages
//
// id ==     "ROOT-ClassName" where ClassName is the affected class
//      else "ROOT/ClassName" where ClassName is the error-declaring class
//      else "ROOT"

  std::string el_identifier = "ROOT" ;

  std::string precursor("class ") ;
  size_t index1 = el_message.find(precursor) ;
  if (index1 != std::string::npos)
    { size_t index2 = index1 + precursor.length() ;
      size_t index3 = el_message.find_first_of(" :", index2) ;
      if (index3 != std::string::npos)
        { size_t substrlen = index3-index2 ;
          el_identifier += "-" ;
          el_identifier += el_message.substr(index2,substrlen) ;
        }
    }
  else
    { index1 = el_location.find("::") ;
      if (index1 != std::string::npos)
        { el_identifier += "/" ;
          el_identifier += el_location.substr(0, index1) ;
        }
    }

// Intercept "dictionary not found" error-level message and alter its severity

  if (el_severity >= edm::ELseverityLevel::ELsev_error)
    { if (el_message.find("dictionary") != std::string::npos)
        { el_severity = edm::ELseverityLevel::ELsev_warning ;
        }
    }

// Feed the message to the MessageLogger... let it choose to suppress or not.

  if ( el_severity == edm::ELseverityLevel::ELsev_fatal )
    {
      edm::LogError("Root_Fatal") << el_location << el_message;
    }
  else if ( el_severity == edm::ELseverityLevel::ELsev_severe )
    {
      edm::LogError("Root_Severe") << el_location << el_message;
    }
  else if ( el_severity == edm::ELseverityLevel::ELsev_error )
    { 
      edm::LogError("Root_Error") << el_location << el_message;
    }
  else if ( el_severity == edm::ELseverityLevel::ELsev_warning )
    {
      edm::LogWarning("Root_Warning") << el_location << el_message ; 
    }
  else if ( el_severity == edm::ELseverityLevel::ELsev_info )
    {
      edm::LogInfo("Root_Information") << el_location << el_message ; 
    }

// Abort, if requested

   if (die)
     { std::cerr << "edm::rootErrorHandler: Aborting as directed. Bye!\n" ;
       ::abort() ;
     }
}
}  // end of unnamed namespace

namespace edm {
namespace service {
InitRootHandlers::InitRootHandlers (edm::ParameterSet const& pset, edm::ActivityRegistry  & activity)
  : unloadSigHandler(pset.getUntrackedParameter<bool> ("UnloadRootSigHandler", false)),
    resetErrHandler (pset.getUntrackedParameter<bool> ("ResetRootErrHandler",  true))
{ 
 
  if( unloadSigHandler ) {
  // Deactivate all the Root signal handlers and restore the system defaults
     edm::LogWarning("Startup") << "@SUB=InitRootHandlers"
                             << "Unload Root signal handlers and restore system defaults" ;
     gSystem->ResetSignal(kSigChild);
     gSystem->ResetSignal(kSigBus);
     gSystem->ResetSignal(kSigSegmentationViolation);
     gSystem->ResetSignal(kSigIllegalInstruction);
     gSystem->ResetSignal(kSigSystem);
     gSystem->ResetSignal(kSigPipe);
     gSystem->ResetSignal(kSigAlarm);
     gSystem->ResetSignal(kSigUrgent);
     gSystem->ResetSignal(kSigFloatingException);
     gSystem->ResetSignal(kSigWindowChanged);
  }

  if( resetErrHandler ) {

  // Replace the Root error handler with one that uses the MessageLogger
     edm::LogWarning("Startup") << "@SUB=InitRootHandlers"
                             << "Install CMS Root error handler" ;
     SetErrorHandler(rootErrorHandler);
  }
}

InitRootHandlers::~InitRootHandlers () {}

}  // end of namespace service
}  // end of namespace edm
