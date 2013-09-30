#ifndef FWCore_MessageService_MessageLogger_h
#define FWCore_MessageService_MessageLogger_h

// -*- C++ -*-
//
// Package:     Services
// Class  :     MessageLogger
//
/**\class MessageLogger MessageLogger.h FWCore/MessageService/interface/MessageLogger.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  W. Brown and M. Fischler
//         Created:  Fri Nov 11 16:38:19 CST 2005
//     Major Split:  Tue Feb 14 15:00:00 CST 2006
//			See FWCore/MessageLogger/MessageLogger.h
//

// system include files

#include <memory>
#include <string>
#include <set>
#include <map>
#include <vector>

// user include files

#include "FWCore/MessageLogger/interface/ErrorObj.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"

// forward declarations

namespace edm  {
class ModuleDescription;
class ParameterSet;
namespace service  {


class MessageLogger {
public:
  MessageLogger( ParameterSet const &, ActivityRegistry & );

  void  fillErrorObj(edm::ErrorObj& obj) const;
  bool  debugEnabled() const { return debugEnabled_; }

  static 
  bool  anyDebugEnabled() { return anyDebugEnabled_; }

  static
  void  SummarizeInJobReport();
  
private:

  void  postBeginJob();
  void  postEndJob();
  void  jobFailure();
  
  void  preSource  ();
  void  postSource ();
  
  void  preFile       ( std::string const &, bool );
  void  preFileClose  ( std::string const&, bool );
  void  postFile      ( std::string const &, bool );
  
  void  preModuleConstruction ( ModuleDescription const & );
  void  postModuleConstruction( ModuleDescription const & );
  
  void  preSourceConstruction ( ModuleDescription const & );
  void  postSourceConstruction( ModuleDescription const & );
  
  void  preModuleEvent ( StreamContext const&, ModuleCallingContext const& );
  void  postModuleEvent( StreamContext const&, ModuleCallingContext const& );
  
  void  preModuleBeginJob  ( ModuleDescription const & );
  void  postModuleBeginJob ( ModuleDescription const & );
  void  preModuleEndJob  ( ModuleDescription const & );
  void  postModuleEndJob ( ModuleDescription const & );

  void  preModuleBeginStream  ( StreamContext const&, ModuleCallingContext const& );
  void  postModuleBeginStream ( StreamContext const&, ModuleCallingContext const& );
  void  preModuleEndStream  ( StreamContext const&, ModuleCallingContext const& );
  void  postModuleEndStream ( StreamContext const&, ModuleCallingContext const& );

  void  preModuleStreamBeginRun  ( StreamContext const&, ModuleCallingContext const& );
  void  postModuleStreamBeginRun ( StreamContext const&, ModuleCallingContext const& );
  void  preModuleStreamEndRun  ( StreamContext const&, ModuleCallingContext const& );
  void  postModuleStreamEndRun ( StreamContext const&, ModuleCallingContext const& );
  
  void  preModuleStreamBeginLumi  ( StreamContext const&, ModuleCallingContext const& );
  void  postModuleStreamBeginLumi ( StreamContext const&, ModuleCallingContext const& );
  void  preModuleStreamEndLumi  ( StreamContext const&, ModuleCallingContext const& );
  void  postModuleStreamEndLumi ( StreamContext const&, ModuleCallingContext const& );
  
  void  preEvent ( StreamContext const& );
  void  postEvent( StreamContext const& );
  
  void  preStreamBeginRun    ( StreamContext const& );
  void  postStreamBeginRun   ( StreamContext const& );
  void  preStreamEndRun      ( StreamContext const& );
  void  postStreamEndRun     ( StreamContext const& );
  void  preStreamBeginLumi   ( StreamContext const& );
  void  postStreamBeginLumi  ( StreamContext const& );
  void  preStreamEndLumi     ( StreamContext const& );
  void  postStreamEndLumi    ( StreamContext const& );

  void  preModuleGlobalBeginRun  ( GlobalContext const&, ModuleCallingContext const& );
  void  postModuleGlobalBeginRun ( GlobalContext const&, ModuleCallingContext const& );
  void  preModuleGlobalEndRun    ( GlobalContext const&, ModuleCallingContext const& );
  void  postModuleGlobalEndRun   ( GlobalContext const&, ModuleCallingContext const& );
  
  void  preModuleGlobalBeginLumi  ( GlobalContext const&, ModuleCallingContext const& );
  void  postModuleGlobalBeginLumi ( GlobalContext const&, ModuleCallingContext const& );
  void  preModuleGlobalEndLumi    ( GlobalContext const&, ModuleCallingContext const& );
  void  postModuleGlobalEndLumi   ( GlobalContext const&, ModuleCallingContext const& );

  void  preGlobalBeginRun    ( GlobalContext const& );
  void  postGlobalBeginRun   ( GlobalContext const& );
  void  preGlobalEndRun      ( GlobalContext const& );
  void  postGlobalEndRun     ( GlobalContext const& );
  void  preGlobalBeginLumi   ( GlobalContext const& );
  void  postGlobalBeginLumi  ( GlobalContext const& );
  void  preGlobalEndLumi     ( GlobalContext const& );
  void  postGlobalEndLumi    ( GlobalContext const& );
  
  void  prePathEvent    ( StreamContext const&, PathContext const& );
  void  postPathEvent   ( StreamContext const&, PathContext const&, HLTPathStatus const&);

  // set up the module name in the message drop, and the enable/suppress info
  void  establishModule       ( const ModuleDescription& desc,
  		                const char* whichPhase );
  void  unEstablishModule     ( const ModuleDescription& desc,
  		                const char*  whichPhase );
  void  establishModule       (unsigned int transitionIndex,
                               const ModuleCallingContext& context,
                               const char* whichPhase );
  void  unEstablishModule     (const ModuleCallingContext& desc,
                               const char*  whichPhase );
  void  establish             ( const char*  whichPhase );
  void  unEstablish           ( const char*  whichPhase ); 
  
  //Cache string description for each active transition
  // stream info is first in the container
  // concurrent lumi info is next
  // concurrent run info is last
  std::vector<std::string> transitionInfoCache_;
  unsigned int lumiInfoBegin_=0;
  unsigned int runInfoBegin_=0;

  std::set<std::string> debugEnabledModules_;
  std::map<std::string,ELseverityLevel> suppression_levels_;
  bool debugEnabled_;
  static bool   anyDebugEnabled_;
  static bool everyDebugEnabled_;

  static bool fjrSummaryRequested_;
  bool messageServicePSetHasBeenValidated_;
  std::string  messageServicePSetValidatationResults_;

  bool nonModule_debugEnabled;
  bool nonModule_infoEnabled;
  bool nonModule_warningEnabled;  
  bool nonModule_errorEnabled;  

};  // MessageLogger

  inline
  bool isProcessWideService(MessageLogger const*) {
    return true;
  }

}  // namespace service

}  // namespace edm



#endif  // FWCore_MessageService_MessageLogger_h

