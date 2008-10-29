#ifndef FWCore_MessageService_MessageLoggerScribe_h
#define FWCore_MessageService_MessageLoggerScribe_h

#include "FWCore/Utilities/interface/value_ptr.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/MessageService/interface/ELdestControl.h"
#include "FWCore/MessageService/interface/MsgContext.h"
#include "FWCore/MessageService/interface/NamedDestination.h"
#include "FWCore/MessageService/interface/MessageLoggerDefaults.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/AbstractMLscribe.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iosfwd>
#include <vector>
#include <map>

#include <iostream>

namespace edm {
namespace service {       

// ----------------------------------------------------------------------
//
// MessageLoggerScribe.h
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
//   2 - ?/?/? before 3/1/07 jm
//	 Modify parameter getting template so as not to catch tracked ones
//	 (that is, to crash if user mistakenly does not say untracked)
//
//   3 - 3/13/07 mf
//	 Added configure_ordinary_destinations, configure_fwkJobReports,
//	 and configure_statistics to allow these to be broken out of 
//	 configure_errorlog. 	
//
//   4 - 3/26/07 mf
//	 Added configure_default_fwkJobReport, which implements the config
//	 originally placed in the .cfi file.
//
//   5 - 6/15/07 mf
//	 Accommodations for use of MessageLoggerDefault structure 
//
//   6 - 7/24/07 mf
//	 Instace variable indicating that we really are logging what is sent.
//	 This is to be able to supress the <generator> info sent at exit by
//	 the JobReport, in the case where no .cfg file was given.
//
//   7 - 4/8/08 mf
//	 Modified getAparameter behavior when tracked parameter is found,
//	 for nicer output message. 
//
//   8 - 6/19/08 mf
//	 triggerFJRmessageSummary
//
//   9 - 10/21/08 mf
//	 variables and routines in preparation for single-thread
//
//  10 - 10/22/08 mf
//	 derivation from AbstractMLscribe to allow for single-thread calling
//	 from MessageLoggerQ without introducing coupling to MessageService
//
// -----------------------------------------------------------------------

class MessageLoggerScribe : public AbstractMLscribe
{
public:
  // ---  birth/death:
  MessageLoggerScribe();
  explicit MessageLoggerScribe(bool singleThreadMode);
  virtual ~MessageLoggerScribe();

  // --- receive and act on messages:
  virtual
  void  run();
  virtual							// changelog 10
  void  runCommand(MessageLoggerQ::OpCode  opcode, void * operand);
		  						// changeLog 9

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
  void triggerFJRmessageSummary(std::map<std::string, double> & sm);
  
  // --- handle details of configuring via a ParameterSet:
  void  configure_errorlog( );
  void  configure_fwkJobReports( );				// Change Log 3
  void  configure_ordinary_destinations( );			// Change Log 3
  void  configure_statistics( );				// Change Log 3
  void  configure_dest( ELdestControl & dest_ctrl		
                      , String const &  filename
		      );
  void  configure_default_fwkJobReport( ELdestControl & dest_ctrl); //ChangeLog 4
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
#ifdef SIMPLESTYLE
  template <class T>
  T  getAparameter ( PSet * p, std::string const & id, T const & def ) 
  {
    T t;
    t = p->template getUntrackedParameter<T>(id, def);
    return t;
  }								// changelog 2
#else
  template <class T>
  T  getAparameter ( PSet * p, std::string const & id, T const & def ) 
  {								// changelog 7
    T t;
    try { 
      t = p->template getUntrackedParameter<T>(id, def);
    } catch (cms::Exception& e) {
      try {
        t = p->template getParameter<T>(id);
      } catch (...) {
        // if we get here, this was NOT a simple tracked parameter goof.
	// we should just rethrow the ORIGINAL error
	throw e;
      } 
      std::cerr << "Tracked parameter " << id 
                << " used in MessageLogger configuration.\n"
		<< "Use of tracked parameters for the message service "
		<< "is not allowed.\n"
		<< "The .cfg file should be modified to make this untracked.\n";
      throw e;		
    } catch (...) {
      // This is not the usual tracked/untracked error; we can't add useful info
      throw;
    }
    return t;
  }



#endif
#endif

#ifdef REMOVE
  template <class T>
  T  getCategoryDefault ( PSet * p, 
  			  std::string const & id, 
			  std::string const & def  ) 
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
  ELadministrator                   * admin_p;
  ELdestControl                       early_dest;
  ErrorLog                          * errorlog_p;
  std::vector<std::ofstream        *> file_ps;
  MsgContext                          msg_context;
  PSet *                              job_pset_p;
  std::vector<NamedDestination     *> extern_dests;
  std::map<String,std::ostream     *> stream_ps;
  std::vector<String> 	  	      ordinary_destination_filenames;
  std::vector<ELdestControl>          statisticsDestControls;
  std::vector<bool>                   statisticsResets;
  std::string	  		      jobReportOption;
  static ErrorLog		    * static_errorlog_p;
  bool				      clean_slate_configuration;
  value_ptr<MessageLoggerDefaults>    messageLoggerDefaults;
  bool				      active;
  bool 				      singleThread;		// changeLog 9
  bool 				      done;			// changeLog 9
  bool 				      purge_mode;		// changeLog 9
  int				      count;			// changeLog 9
      
};  // MessageLoggerScribe


}   // end of namespace service
}  // namespace edm


#endif  // FWCore_MessageService_MessageLoggerScribe_h
