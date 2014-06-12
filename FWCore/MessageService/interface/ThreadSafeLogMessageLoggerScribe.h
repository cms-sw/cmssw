#ifndef FWCore_MessageService_ThreadSafeLogMessageLoggerScribe_h
#define FWCore_MessageService_ThreadSafeLogMessageLoggerScribe_h

#include "FWCore/Utilities/interface/value_ptr.h"

#include "FWCore/MessageService/interface/ELdestControl.h"
#include "FWCore/MessageService/interface/MessageLoggerDefaults.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/AbstractMLscribe.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

#include <iosfwd>
#include <vector>
#include <map>

#include <iostream>
#include <atomic>
#include "tbb/concurrent_queue.h"

namespace edm {
namespace service {       

// ----------------------------------------------------------------------
//
// ThreadSafeLogMessageLoggerScribe.h
//
// OpCodeLOG_A_MESSAGE messages can be handled from multiple threads
//
// -----------------------------------------------------------------------

class ThreadQueue;
class ELadministrator;

class ThreadSafeLogMessageLoggerScribe : public AbstractMLscribe
{
public:
  // ---  birth/death:
  
  // ChangeLog 12
  /// --- If queue is NULL, this sets singleThread true 
  explicit ThreadSafeLogMessageLoggerScribe();
  
  virtual ~ThreadSafeLogMessageLoggerScribe();

  // --- receive and act on messages:
  virtual							// changelog 10
  void  runCommand(MessageLoggerQ::OpCode  opcode, void * operand);
		  						// changeLog 9

private:
  // --- convenience typedefs
  typedef std::string          String;
  typedef std::vector<String>  vString;
  typedef ParameterSet         PSet;

  // --- log one consumed message
  void log(ErrorObj * errorobj_p);

  // --- cause statistics destinations to output
  void triggerStatisticsSummaries();
  void triggerFJRmessageSummary(std::map<std::string, double> & sm);
  
  // --- handle details of configuring via a ParameterSet:
  void  configure_errorlog( );
  void  configure_ordinary_destinations( );			// Change Log 3
  void  configure_statistics( );				// Change Log 3
  void  configure_dest( ELdestControl & dest_ctrl		
                      , String const &  filename
		      );
  void  configure_external_dests( );

  template <class T>						// ChangeLog 11
  T getAparameter ( PSet const& p, std::string const & id, T const & def ) 
  {
    T t = def;
    try { 
      t = p.template getUntrackedParameter<T>(id, def);
    } catch (...) {
      try {
        t = p.template getParameter<T>(id);
      } catch (...) {
        // Since PSetValidation will catch such errors, we simply proceed as
	// best we can in case we are setting up the logger just to contain the 
	// validation-caught error messages. 
      }
    }
    return t;
  }

  // --- other helpers
  void parseCategories (std::string const & s, std::vector<std::string> & cats);
  
  // --- data:
  std::shared_ptr<ELadministrator>  admin_p;
  ELdestControl                       early_dest;
  std::vector<std::shared_ptr<std::ofstream> > file_ps;
  std::shared_ptr<PSet>             job_pset_p;
  std::vector<NamedDestination     *> extern_dests;
  std::map<String,std::ostream     *> stream_ps;
  std::vector<String> 	  	      ordinary_destination_filenames;
  std::vector<ELdestControl>          statisticsDestControls;
  std::vector<bool>                   statisticsResets;
  bool				      clean_slate_configuration;
  value_ptr<MessageLoggerDefaults>    messageLoggerDefaults;
  bool				      active;
  std::atomic<bool> purge_mode;		// changeLog 9
  std::atomic<int>  count;			// changeLog 9
  std::atomic<bool> m_messageBeingSent;
  tbb::concurrent_queue<ErrorObj*> m_waitingMessages;
  
};  // ThreadSafeLogMessageLoggerScribe


}   // end of namespace service
}  // namespace edm


#endif  // FWCore_MessageService_ThreadSafeLogMessageLoggerScribe_h
