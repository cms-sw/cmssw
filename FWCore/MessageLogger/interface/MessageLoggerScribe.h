#ifndef FWCore_MessageLogger_MessageLoggerScribe_h
#define FWCore_MessageLogger_MessageLoggerScribe_h

#include "FWCore/MessageLogger/interface/ELadministrator.h"
#include "FWCore/MessageLogger/interface/ELdestControl.h"
#include "FWCore/MessageLogger/interface/ErrorLog.h"
#include "FWCore/MessageLogger/interface/MsgContext.h"
#include "FWCore/MessageLogger/interface/NamedDestination.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <fstream>
#include <vector>


namespace edm
{


class MessageLoggerScribe
{
public:
  // ---  birth/death:
  MessageLoggerScribe();
  ~MessageLoggerScribe();

  // --- receive and act on messages:
  void  run();

private:
  // --- convenience typedefs
  typedef std::string          String;
  typedef std::vector<String>  vString;
  typedef ParameterSet         PSet;

  // --- log one consumed message
  void log(ErrorObj * errorobj_p);

  // --- handle details of configuring via a ParameterSet:
  void  configure_errorlog( );
  void  configure_dest( ELdestControl & dest_ctrl
                      , String const &  filename
		      );
  void  configure_external_dests( );

  // --- other helpers
  void parseCategories (std::string const & s, std::vector<std::string> & cats);
  
  // --- data:
  ELadministrator               * admin_p;
  ELdestControl                   early_dest;
  ErrorLog                      * errorlog_p;
  std::vector<std::ofstream    *> file_ps;
  edm::MsgContext                 msg_context;
  PSet *                          job_pset_p;
  std::vector<NamedDestination *> extern_dests;

};  // MessageLoggerScribe


}  // namespace edm


#endif  // FWCore_MessageLogger_MessageLoggerScribe_h
