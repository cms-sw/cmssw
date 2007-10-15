#ifndef FWCore_MessageService_MessageLoggerScribe_h
#define FWCore_MessageService_MessageLoggerScribe_h

#include "FWCore/MessageService/interface/ELadministrator.h"
#include "FWCore/MessageService/interface/ELdestControl.h"
#include "FWCore/MessageService/interface/ErrorLog.h"
#include "FWCore/MessageService/interface/MsgContext.h"
#include "FWCore/MessageService/interface/NamedDestination.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>


namespace edm {
namespace service {       

// ----------------------------------------------------------------------
//
// MessageLoggerScribe.cc
//
// Changes:
//
//   1 - 2/6/07  mf  	
//	Set up ability to get a remembered pointer to the ErrorLog of an
//      instance of MessageLoggerScribe, from a non-member function, via
//	getErrorLog_ptr(), and a corresponding routine to remember the pointer
//	as setStaticErrorLog_ptr().  Needed if we decide to send an explicit 
//      message from the scribe.
//	
// -----------------------------------------------------------------------

class MessageLoggerScribe
{
public:
  // ---  birth/death:
  MessageLoggerScribe();
  ~MessageLoggerScribe();

  // --- receive and act on messages:
  void  run();

  // --- obtain a pointer to the errorlog 
  static ErrorLog * getErrorLog_ptr() {return static_errorlog_p;}
  
private:
  // --- convenience typedefs
  typedef std::string          String;
  typedef std::vector<String>  vString;
  typedef ParameterSet         PSet;

  // --- log one consumed message
  void log(ErrorObj * errorobj_p);

  // --- cause statistics destinations to output
  void triggerStatisticsSummaries();

  // --- handle details of configuring via a ParameterSet:
  void  configure_errorlog( );
  void  configure_dest( ELdestControl & dest_ctrl
                      , String const &  filename
		      );
  void  configure_external_dests( );

#ifdef OLDSTYLE
  template <class T>
  T  getAparameter ( PSet * p, std::string const & id, T const & def ) 
  {
    T t;
    try { 
      t = p->template getUntrackedParameter<T>(id, def);
    } catch (...) {
      t = p->template getParameter<T>(id);
      std::cerr << "Tracked parameter " << id 
                << " used in MessageLogger configuration.\n"
		<< "Use of tracked parameters for the message service "
		<< "is deprecated.\n"
		<< "The .cfg file should be modified to make this untracked.\n";
    }
    return t;
  }
#else
  template <class T>
  T  getAparameter ( PSet * p, std::string const & id, T const & def ) 
  {
    T t;
    t = p->template getUntrackedParameter<T>(id, def);
    return t;
  }
#endif

  // --- other helpers
  void parseCategories (std::string const & s, std::vector<std::string> & cats);
  void setStaticErrorLog_ptr() {static_errorlog_p = errorlog_p;}
  
  // --- data:
  ELadministrator               * admin_p;
  ELdestControl                   early_dest;
  ErrorLog                      * errorlog_p;
  std::vector<std::ofstream    *> file_ps;
  MsgContext                      msg_context;
  PSet *                          job_pset_p;
  std::vector<NamedDestination *> extern_dests;
  std::map<String,std::ostream *> stream_ps;
  std::vector<ELdestControl>      statisticsDestControls;
  std::vector<bool>               statisticsResets;
  static ErrorLog		* static_errorlog_p;
};  // MessageLoggerScribe


}   // end of namespace service
}  // namespace edm


#endif  // FWCore_MessageService_MessageLoggerScribe_h
