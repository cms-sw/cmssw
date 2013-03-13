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

  void  postBeginJob();
  void  postEndJob();
  void  jobFailure();
  
  void  preSource  ();
  void  postSource ();

  void  preFile       ();
  void  preFileClose  ( std::string const&, bool );
  void  postFile      ();

  void  preModuleConstruction ( ModuleDescription const & );
  void  postModuleConstruction( ModuleDescription const & );

  void  preSourceConstruction ( ModuleDescription const & );
  void  postSourceConstruction( ModuleDescription const & );

  void  preModule ( ModuleDescription const & );
  void  postModule( ModuleDescription const & );

  void  preModuleBeginJob  ( ModuleDescription const & );
  void  postModuleBeginJob ( ModuleDescription const & );
  void  preModuleEndJob  ( ModuleDescription const & );
  void  postModuleEndJob ( ModuleDescription const & );

  void  preModuleBeginRun  ( ModuleDescription const & );
  void  postModuleBeginRun ( ModuleDescription const & );
  void  preModuleEndRun  ( ModuleDescription const & );
  void  postModuleEndRun ( ModuleDescription const & );

  void  preModuleBeginLumi  ( ModuleDescription const & );
  void  postModuleBeginLumi ( ModuleDescription const & );
  void  preModuleEndLumi  ( ModuleDescription const & );
  void  postModuleEndLumi ( ModuleDescription const & );

  void  preEventProcessing ( edm::EventID const &, edm::Timestamp const & );
  void  postEventProcessing( Event const &, EventSetup const & );

  void  preBeginRun    ( const edm::RunID&, const edm::Timestamp& );
  void  postBeginRun   ( const edm::Run&, EventSetup const & );
  void  preEndRun      ( const edm::RunID&, const edm::Timestamp& );
  void  postEndRun     ( const edm::Run&, EventSetup const & );
  void  preBeginLumi   ( const edm::LuminosityBlockID&, const edm::Timestamp& );
  void  postBeginLumi  ( const edm::LuminosityBlock&, EventSetup const & );
  void  preEndLumi     ( const edm::LuminosityBlockID&, const edm::Timestamp& );
  void  postEndLumi    ( const edm::LuminosityBlock&, EventSetup const & );

  void  prePathBeginRun   ( const std::string& pathname );
  void  postPathBeginRun  ( const std::string& pathname, HLTPathStatus const&);
  void  prePathEndRun     ( const std::string& pathname );
  void  postPathEndRun    ( const std::string& pathname, HLTPathStatus const&);
  void  prePathBeginLumi  ( const std::string& pathname );
  void  postPathBeginLumi ( const std::string& pathname, HLTPathStatus const&);
  void  prePathEndLumi    ( const std::string& pathname );
  void  postPathEndLumi   ( const std::string& pathname, HLTPathStatus const&);
  void  preProcessPath    ( const std::string& pathname );
  void  postProcessPath   ( const std::string& pathname, HLTPathStatus const&);


  void  fillErrorObj(edm::ErrorObj& obj) const;
  bool  debugEnabled() const { return debugEnabled_; }

  static 
  bool  anyDebugEnabled() { return anyDebugEnabled_; }

  static
  void  SummarizeInJobReport();
  
private:

  // set up the module name in the message drop, and the enable/suppress info
  void  establishModule       ( const ModuleDescription& desc,
  		                const char* whichPhase );
  void  establishModuleCtor   ( const ModuleDescription& desc,
  		                const char*  whichPhase );
  void  unEstablishModule     ( const ModuleDescription& desc,
  		                const char*  whichPhase );
  void  establish             ( const char*  whichPhase ); 
  void  unEstablish           ( const char*  whichPhase ); 
  
  edm::EventID curr_event_;

  std::set<std::string> debugEnabledModules_;
  std::map<std::string,ELseverityLevel> suppression_levels_;
  bool debugEnabled_;
  //this is a cache which profiling has shown to be helpful
  std::map<const ModuleDescription*, std::string> descToCalcName_;
  static bool   anyDebugEnabled_;
  static bool everyDebugEnabled_;

  static bool fjrSummaryRequested_;
  bool messageServicePSetHasBeenValidated_;
  std::string  messageServicePSetValidatationResults_;

  bool nonModule_debugEnabled;
  bool nonModule_infoEnabled;
  bool nonModule_warningEnabled;  
  bool nonModule_errorEnabled;  

  std::vector<ModuleDescription const*> moduleStack_;

};  // MessageLogger

  inline
  bool isProcessWideService(MessageLogger const*) {
    return true;
  }

}  // namespace service

}  // namespace edm



#endif  // FWCore_MessageService_MessageLogger_h

