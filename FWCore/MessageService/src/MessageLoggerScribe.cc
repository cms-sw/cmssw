// ----------------------------------------------------------------------
//
// MessageLoggerScribe.cc
//
// Changes:
//
//   1 - 3/22/06  mf  - in configure_dest()	
//	Repaired the fact that destination limits for categories
//	were not being effective:
//	a) use values from the destination specific default PSet
//	   rather than the overall default PSet to set these
//	b) when an explicit value has been set - either by overall default or 
//	   by a destination specific default PSet - set that limit or
//	   timespan for that dest_ctrl via a "*" msgId.
// 
//   2 - 3/22/06  mf  - in configure_dest()	
//	Enabled the use of -1 in the .cfg file to mean infinite limit
//	or timespan.  This is done by:
//	a) replacing the default value of -1 (by which we recognize 
//	never-specified values) by NO_VALUE_SET = -45654
//	b) checking for values of -1 and substituting a very large integer  
//
//   3 - 4/28/06  mf  - in configure_dest()
//	Mods to help deal with the fact that checking for an empty PSet is
//	unwise when untracked parameters are involved:  The PSet will appear
//	to be empty and if skipped, will result in limits not being applied.
//	a) Replaced default values directly in getAparameter with variables
//	which can be examined all in one place.
//	b) Carefully checked that we are never comparing to the empty PSet
//	
//   4 - 4/28/06  mf  - in configure_dest()
//	If a destination name does not have an extension, append .log 
//	(or in the case of a FwkJobReport, .xml).
//	[note for this change - the filename kept as an index to stream_ps
//	can be kept as its original name; it is just a tool for assigning
//	the right shared stream to statistics destinations]
//
//   5 - 4/28/06  mf  - in configure_dest()
//	Provision for an overall default affecting all categories, for 
//	example, establishing a limit for all a specific category for
//	every destination. 
//
//   6 - 5/18/06 mf  - in configure_dest()
//	Implement establishing intervals between reacting to message of
//	some type.
//
//   7 - 5/24/06 mf  - in configure_dest()
//	Corrected algorithm for estabolishing limits and intervals, avoiding
//      interference between setting the one and getting the default for the 
//      other.
//
//   8 - 5/31/06 wmtan  - in configure_errorlog()
//	The presence of the framework job report should not affect the output
//      to the early destination (cerr).
//
//   9 - 6/6/06 mf  - in configure_dest()
//	Support for placeholder PSet without actually creating the destination.
//	Useful in a .cfi file, in conjunction with potential replace commands.
//
//  10 - 6/6/06 mf  - in configure_dest()
//	Changed cfg keyward interval to reportEvery
//
//  11 - 6/12/06 mf  - in configure_errorlog()
//	Check for placeholder before attaching a destination that may not be
//	wanted.
//
//  12 - 6/14/06 mf  - in configure_external_dests()
//	Clear the list of external dests needing configuration, if there
//	is no configuration file available.
//
//  13 - 8/7/06 mf  - in configure_external_dests()
//	Undo change 12: the list of external dests needing configuration
//      is left intact if there is no configuration file available, the
//      assumption being that at some later time there will be a file and
//      the message logger will be configured again.
//
//      Note: The change made in (12) and un-done here was necessary to
//            prevent segfault behavior when a job is done with external
//            destinations and no .cfg file under some circumstances.
//            D. Evans (who was being hit with that behavior due to an
//	      accidental .cfg omission) asserts (8/16) that running with
//            no .cfg file is a sufficient anomoly that the current change
//            is acceptable. 
//
//  14 - 10/18/06 mf  - in configure_error_log()
//	Finer control of output file name for a given destination:
//	Accept a parameter extension, to specify some extension other than
//      .log without needing to place a dot in the Pset name.  Also accept
//	an explicit filename.
//
// ----------------------------------------------------------------------



#include "FWCore/MessageService/interface/ELoutput.h"
#include "FWCore/MessageService/interface/ELstatistics.h"
#include "FWCore/MessageService/interface/ELfwkJobReport.h"
#include "FWCore/MessageService/interface/MessageLogger.h"
#include "FWCore/MessageService/interface/MessageLoggerScribe.h"
#include "FWCore/MessageService/interface/NamedDestination.h"

#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using std::cerr;

namespace edm {
namespace service {


MessageLoggerScribe::MessageLoggerScribe()
: admin_p   ( ELadministrator::instance() )
, early_dest( admin_p->attach(ELoutput(std::cerr, false)) )
, errorlog_p( new ErrorLog() )
, file_ps   ( )
, job_pset_p( 0 )
, extern_dests( )
{
  admin_p->setContextSupplier(msg_context);
}


MessageLoggerScribe::~MessageLoggerScribe()
{
  admin_p->finish();
  delete errorlog_p;
  for( ;  not file_ps.empty();  file_ps.pop_back() )  {
    delete file_ps.back();
  }
  delete job_pset_p; // dispose of our (copy of the) ParameterSet
  assert( extern_dests.empty() );  // nothing to do
}


void
  MessageLoggerScribe::run()
{
  MessageLoggerQ::OpCode  opcode;
  void *                  operand;
  bool  done = false;
  bool  purge_mode = false;
  int count = 0;

  do  {
    MessageLoggerQ::consume(opcode, operand);  // grab next work item from Q
    switch(opcode)  {  // interpret the work item
      default:  {
        assert(false);  // can't happen (we certainly hope!)
        break;
      }
      case MessageLoggerQ::END_THREAD:  {
        assert( operand == 0 );
        done = true;
        break;
      }
      case MessageLoggerQ::LOG_A_MESSAGE:  {
        ErrorObj *  errorobj_p = static_cast<ErrorObj *>(operand);
	try {
	  if(!purge_mode) log (errorobj_p);        
	}
	catch(cms::Exception& e)
	  {
	    ++count;
	    cerr << "MessageLoggerScribe caught " << count
		 << " cms::Exceptions, text = \n"
		 << e.what() << "\n";
	    
	    if(count > 5)
	      {
		cerr << "MessageLogger will no longer be processing "
		     << "messages due to errors. (entering purge mode)\n";
		purge_mode = true;
	      }
	  }
	catch(...)
	  {
	    cerr << "MessageLoggerScribe caught an unknown exception and "
		 << "will no longer be processing "
		 << "messages. (entering purge mode)\n";
	    purge_mode = true;
	  }
        delete errorobj_p;  // dispose of the message text
        break;
      }
      case MessageLoggerQ::CONFIGURE:  {
	try {
	  job_pset_p = static_cast<PSet *>(operand);
	  configure_errorlog();
	}
	catch(cms::Exception& e)
	  {
	    cerr << "MessageLoggerScribe caught exception "
		 << "during configuration:\n"
		 << e.what() << "\n"
		 << "Aborting the job with return code -5.\n";
	    exit(-5);
	  }
	catch(...)
	  {
	    cerr << "MessageLoggerScribe caught unkonwn exception type\n"
		 << "during configuration. "
		 << "Aborting the job with return code -5.\n";
	    exit(-5);
	  }
        break;
      }
      case MessageLoggerQ::EXTERN_DEST: {
	try {
	  extern_dests.push_back( static_cast<NamedDestination *>(operand) );
	  configure_external_dests();
	}
	catch(cms::Exception& e)
	  {
	    cerr << "MessageLoggerScribe caught exception "
		 << "during extern dest configuration:\n"
		 << e.what() << "\n"
		 << "Aborting the job with return code -5.\n";
	    exit(-5);
	  }
	catch(...)
	  {
	    cerr << "MessageLoggerScribe caught unkonwn exception type\n"
		 << "during extern dest configuration. "
		 << "Aborting the job with return code -5.\n";
	    exit(-5);
	  }
        break;
      }
      case MessageLoggerQ::SUMMARIZE: {
        assert( operand == 0 );
	try {
	  triggerStatisticsSummaries();
	}
	catch(cms::Exception& e)
	  {
	    cerr << "MessageLoggerScribe caught exception "
		 << "during summarize:\n"
		 << e.what() << "\n";
	  }
	catch(...)
	  {
	    cerr << "MessageLoggerScribe caught unkonwn exception type "
		 << "during summarize. (Ignored)\n";
	  }
        break;
      }
    }  // switch

  } while(! done);

}  // MessageLoggerScribe::run()

void MessageLoggerScribe::log ( ErrorObj *  errorobj_p ) {
  ELcontextSupplier& cs =
    const_cast<ELcontextSupplier&>(admin_p->getContextSupplier());
  MsgContext& mc = dynamic_cast<MsgContext&>(cs);
  mc.setContext(errorobj_p->context());
  std::vector<std::string> categories;
  parseCategories(errorobj_p->xid().id, categories);
  for (unsigned int icat = 0; icat < categories.size(); ++icat) {
    errorobj_p->setID(categories[icat]);
    (*errorlog_p)( *errorobj_p );  // route the message text
  } 
}

void
  MessageLoggerScribe::configure_errorlog()
{
  vString  empty_vString;
  String   empty_String;
  PSet     empty_PSet;
  
  // The following is present to test pre-configuration message handling:
  String preconfiguration_message 
       = getAparameter<String>
       	(job_pset_p, "generate_preconfiguration_message", empty_String);
  if (preconfiguration_message != empty_String) {
    // To test a preconfiguration message without first going thru the 
    // configuration we are about to do, we issue the message (so it sits
    // on the queue), then copy the processing that the LOG_A_MESSAE case
    // does.  We suppress the timestamp to allow for automated unit testing.
    early_dest.suppressTime();
    LogError ("preconfiguration") << preconfiguration_message;
    MessageLoggerQ::OpCode  opcode;
    void *                  operand;
    MessageLoggerQ::consume(opcode, operand);  // grab next work item from Q
    assert (opcode == MessageLoggerQ::LOG_A_MESSAGE);
    ErrorObj *  errorobj_p = static_cast<ErrorObj *>(operand);
    log (errorobj_p);        
    delete errorobj_p;  // dispose of the message text
  }

  // We will need a map of   
  // grab list of destinations:
  vString  destinations
     = getAparameter<vString>(job_pset_p, "destinations", empty_vString);

  // dial down the early destination if other dest's are supplied:
  if( ! destinations.empty() )
    early_dest.setThreshold(ELhighestSeverity);

  // establish each destination:
  for( vString::const_iterator it = destinations.begin()
     ; it != destinations.end()
     ; ++it
     )
  {
    String filename = *it;
    String psetname = filename;
    
    // check that this destination is not just a placeholder // change log 11
    PSet  dest_pset = getAparameter<PSet>(job_pset_p,psetname,empty_PSet);
    bool is_placeholder 
	= getAparameter<bool>(&dest_pset,"placeholder", false);
    if (is_placeholder) continue;

    // Modify the file name if extension or name is explicitly specified
    // change log 14 
    String explicit_filename 
        = getAparameter<String>(&dest_pset,"filename",empty_String);
    if (explicit_filename != empty_String) filename = explicit_filename;
    String explicit_extension 
        = getAparameter<String>(&dest_pset,"extension",empty_String);
    if (explicit_extension != empty_String) {
      if (explicit_extension[0] == '.') {
	filename += explicit_extension;             
      } else {
	filename = filename + "." + explicit_extension;   
      }
    }

    // attach the current destination, keeping a control handle to it:
    ELdestControl dest_ctrl;
    if( filename == "cout" )  {
      dest_ctrl = admin_p->attach( ELoutput(std::cout) );
      stream_ps["cout"] = &std::cout;
    }
    else if( filename == "cerr" )  {
      early_dest.setThreshold(ELzeroSeverity); 
      dest_ctrl = early_dest;
      stream_ps["cerr"] = &std::cerr;
    }
    else  {
      std::string actual_filename = filename;			// change log 4
      const std::string::size_type npos = std::string::npos;
      if ( filename.find('.') == npos ) {
        actual_filename += ".log";
      }  
      std::ofstream * os_p = new std::ofstream(actual_filename.c_str());
      file_ps.push_back(os_p);
      dest_ctrl = admin_p->attach( ELoutput(*os_p) );
      stream_ps[filename] = os_p;
    }
    //(*errorlog_p)( ELinfo, "added_dest") << filename << endmsg;

    // now configure this destination:
    configure_dest(dest_ctrl, psetname);

  }  // for [it = destinations.begin() to end()]

  // grab list of fwkJobReports:
  vString  fwkJobReports
     = getAparameter<vString>(job_pset_p, "fwkJobReports", empty_vString);

  // dial down the early destination if other dest's are supplied:
  // if( ! fwkJobReports.empty() ) // change log 8
  //   early_dest.setThreshold(ELhighestSeverity);

  // establish each fwkJobReports destination:
  for( vString::const_iterator it = fwkJobReports.begin()
     ; it != fwkJobReports.end()
     ; ++it
     )
  {
    // attach the current destination, keeping a control handle to it:
    ELdestControl dest_ctrl;
    String filename = *it;
    String psetname = filename;

    // Modify the file name if extension or name is explicitly specified
    // change log 14 
    PSet  fjr_pset = getAparameter<PSet>(job_pset_p,psetname,empty_PSet);
    String explicit_filename 
        = getAparameter<String>(&fjr_pset,"filename",empty_String);
    if (explicit_filename != empty_String) filename = explicit_filename;
    String explicit_extension 
        = getAparameter<String>(&fjr_pset,"extension",empty_String);
    if (explicit_extension != empty_String) {
      if (explicit_extension[0] == '.') {
	filename += explicit_extension;             
      } else {
	filename = filename + "." + explicit_extension;   
      }
    }

    std::string actual_filename = filename;			// change log 4
    const std::string::size_type npos = std::string::npos;
    if ( filename.find('.') == npos ) {
      actual_filename += ".xml";
    }  
    std::ofstream * os_p = new std::ofstream(actual_filename.c_str());
    file_ps.push_back(os_p);
    dest_ctrl = admin_p->attach( ELfwkJobReport(*os_p) );
    stream_ps[filename] = os_p;

    // now configure this destination:
    configure_dest(dest_ctrl, psetname);	

  }  // for [it = fwkJobReports.begin() to end()]

  // grab list of statistics destinations:
  vString  statistics 
     = getAparameter<vString>(job_pset_p,"statistics", empty_vString);

   // establish each statistics destination:
  for( vString::const_iterator it = statistics.begin()
     ; it != statistics.end()
     ; ++it
     )
  {
    // determine the filename to be used:
    // either the statistics name or if a Pset by that name has 
    // file = somename, then that specified name.
    String statname = *it;
    String psetname = statname;
    PSet  stat_pset 
    	= getAparameter<PSet>(job_pset_p,psetname,empty_PSet);
    String filename 
        = getAparameter<String>(&stat_pset,"output",statname);
    
    // Modify the file name if extension or name is explicitly specified
    // change log 14 -- probably suspenders and a belt, because ouput option
    // is present, but uniformity is nice.
    String explicit_filename 
        = getAparameter<String>(&stat_pset,"filename",empty_String);
    if (explicit_filename != empty_String) filename = explicit_filename;
    String explicit_extension 
        = getAparameter<String>(&stat_pset,"extension",empty_String);
    if (explicit_extension != empty_String) {
      if (explicit_extension[0] == '.') {
	filename += explicit_extension;             
      } else {
	filename = filename + "." + explicit_extension;   
      }
    }

    // create (if statistics file does not match any destination file name)
    // or note (if statistics file matches a destination file name) the ostream
    std::ostream * os_p;
    if ( stream_ps.find(filename) == stream_ps.end() ) {
      if ( filename == "cout" ) {
        os_p = &std::cout;
      } else if ( filename == "cerr" ) {
        os_p = &std::cerr;
      } else {
        std::string actual_filename = filename;			// change log 4
        const std::string::size_type npos = std::string::npos;
        if ( filename.find('.') == npos ) {
          actual_filename += ".log";
        }  
        std::ofstream * osf_p = new std::ofstream(actual_filename.c_str());
        os_p = osf_p;
	file_ps.push_back(osf_p);
      }
      stream_ps[filename] = os_p;
    } else { 
      os_p = stream_ps[filename];
    }
       
    // attach the statistics destination, keeping a control handle to it:
    ELdestControl dest_ctrl;
    dest_ctrl = admin_p->attach( ELstatistics(*os_p) );
    statisticsDestControls.push_back(dest_ctrl);
    bool reset = getAparameter<bool>(&stat_pset,"reset",false);
    statisticsResets.push_back(reset);

    // now configure this destination:
    configure_dest(dest_ctrl, psetname);

    // and suppress the desire to do an extra termination summary just because
    // of end-of-job info messages
    dest_ctrl.noTerminationSummary();
 
  }  // for [it = statistics.begin() to end()]

  configure_external_dests();

}  // MessageLoggerScribe::configure_errorlog()


void
  MessageLoggerScribe::configure_dest( ELdestControl & dest_ctrl
                                     , String const &  filename
				     )
{
  static const int NO_VALUE_SET = -45654;		// change log 2
  vString  empty_vString;
  PSet     empty_PSet;
  String   empty_String;

  // Defaults:						// change log 3a
  const std::string COMMON_DEFAULT_THRESHOLD = "INFO";
  const         int COMMON_DEFAULT_LIMIT     = NO_VALUE_SET; 
  const         int COMMON_DEFAULT_INTERVAL  = NO_VALUE_SET; // change log 6
  const         int COMMON_DEFAULT_TIMESPAN  = NO_VALUE_SET; 

  char *  severity_array[] = {"WARNING", "INFO", "ERROR", "DEBUG"};
  vString const  severities(severity_array+0, severity_array+4);


  // grab list of categories
  vString  categories
     = getAparameter<vString>(job_pset_p,"categories", empty_vString);

  // grab list of messageIDs -- these are a synonym for categories
  // Note -- the use of messageIDs is deprecated in favor of categories
  {
    vString  messageIDs
      = getAparameter<vString>(job_pset_p,"messageIDs", empty_vString);

  // combine the lists, not caring about possible duplicates (for now)
    std::copy( messageIDs.begin(), messageIDs.end(),
               std::back_inserter(categories)
             );
  }  // no longer need messageIDs

  // grab default threshold common to all destinations
  String default_threshold
     = getAparameter<String>(job_pset_p,"threshold", COMMON_DEFAULT_THRESHOLD);
     						// change log 3a

  // grab default limit/interval/timespan common to all destinations/categories:
  PSet  default_pset
     = getAparameter<PSet>(job_pset_p,"default", empty_PSet);
  int  default_limit
    = getAparameter<int>(&default_pset,"limit", COMMON_DEFAULT_LIMIT);
  int  default_interval
    = getAparameter<int>(&default_pset,"reportEvery", COMMON_DEFAULT_INTERVAL);
    						// change log 6, 10
  int  default_timespan
    = getAparameter<int>(&default_pset,"timespan", COMMON_DEFAULT_TIMESPAN);
						// change log 2a
    						// change log 3a
					
  // grab all of this destination's parameters:
  PSet  dest_pset = getAparameter<PSet>(job_pset_p,filename,empty_PSet);

  // See if this is just a placeholder			// change log 9
  bool is_placeholder 
      = getAparameter<bool>(&dest_pset,"placeholder", false);
  if (is_placeholder) return;
  
  // grab this destination's default limit/interval/timespan:
  PSet  dest_default_pset
     = getAparameter<PSet>(&dest_pset,"default", empty_PSet);
  int  dest_default_limit
    = getAparameter<int>(&dest_default_pset,"limit", default_limit);
  int  dest_default_interval
    = getAparameter<int>(&dest_default_pset,"reportEvery", default_interval);
    						// change log 6
  int  dest_default_timespan
    = getAparameter<int>(&dest_default_pset,"timespan", default_timespan);
    						// change log 1a
  if ( dest_default_limit != NO_VALUE_SET ) {
    if ( dest_default_limit < 0 ) dest_default_limit = 2000000000;
    dest_ctrl.setLimit("*", dest_default_limit );
  } 						// change log 1b, 2a, 2b
  if ( dest_default_interval != NO_VALUE_SET ) {  // change log 6
    dest_ctrl.setInterval("*", dest_default_interval );
  } 						
  if ( dest_default_timespan != NO_VALUE_SET ) {
    if ( dest_default_timespan < 0 ) dest_default_timespan = 2000000000;
    dest_ctrl.setTimespan("*", dest_default_timespan );
  } 						// change log 1b, 2a, 2b
    						  
  // establish this destination's threshold:
  String dest_threshold
     = getAparameter<String>(&dest_pset,"threshold", default_threshold);
  ELseverityLevel  threshold_sev(dest_threshold);
  dest_ctrl.setThreshold(threshold_sev);

  // establish this destination's limit/interval/timespan for each category:
  for( vString::const_iterator id_it = categories.begin()
     ; id_it != categories.end()
     ; ++id_it
     )
  {
    String  msgID = *id_it;
    PSet default_category_pset 
       = getAparameter<PSet>(&default_pset,msgID, empty_PSet);	// change log 5
    PSet  category_pset
       = getAparameter<PSet>(&dest_pset,msgID, default_category_pset);
    int  category_default_limit 
       = getAparameter<int>(&default_category_pset,"limit",NO_VALUE_SET);
    int  limit
      = getAparameter<int>(&category_pset,"limit", category_default_limit);
    if (limit == NO_VALUE_SET) limit = dest_default_limit;
       								// change log 7 
    int  category_default_interval 
       = getAparameter<int>(&default_category_pset,"reportEvery",NO_VALUE_SET);
    int  interval
      = getAparameter<int>(&category_pset,"reportEvery",category_default_interval);
    if (interval == NO_VALUE_SET) interval = dest_default_interval;
      						// change log 6  and then 7
    int  category_default_timespan 
       = getAparameter<int>(&default_category_pset,"timespan",NO_VALUE_SET);
    int  timespan
      = getAparameter<int>(&category_pset,"timespan",category_default_timespan);
    if (timespan == NO_VALUE_SET) timespan = dest_default_timespan;
       								// change log 7 
      
    if( limit     != NO_VALUE_SET )  {
      if ( limit < 0 ) limit = 2000000000;  
      dest_ctrl.setLimit(msgID, limit);
    }  						// change log 2a, 2b
    if( interval  != NO_VALUE_SET )  {
      dest_ctrl.setInterval(msgID, interval);
    }  						// change log 6
    if( timespan  != NO_VALUE_SET )  {
      if ( timespan < 0 ) timespan = 2000000000;  
      dest_ctrl.setTimespan(msgID, timespan);
    }						// change log 2a, 2b
						
  }  // for

  // establish this destination's limit for each severity:
  for( vString::const_iterator sev_it = severities.begin()
     ; sev_it != severities.end()
     ; ++sev_it
     )
  {
    String  sevID = *sev_it;
    ELseverityLevel  severity(sevID);
    PSet  default_sev_pset 
    	= getAparameter<PSet>(&default_pset,sevID, empty_PSet);
    PSet  sev_pset 
    	= getAparameter<PSet>(&dest_pset,sevID, default_sev_pset);
						// change log 5
    int  limit     = getAparameter<int>(&sev_pset,"limit",    NO_VALUE_SET);
    if( limit    != NO_VALUE_SET )  dest_ctrl.setLimit(severity, limit   );
    int  timespan  = getAparameter<int>(&sev_pset,"timespan", NO_VALUE_SET);
    if( timespan != NO_VALUE_SET )  dest_ctrl.setLimit(severity, timespan);
						// change log 2
    int  interval  = getAparameter<int>(&sev_pset,"reportEvery", NO_VALUE_SET);
    if( limit    != NO_VALUE_SET )  dest_ctrl.setInterval(severity, interval   );
    						// change log 6
  }  // for

  // establish this destination's linebreak policy:
  bool noLineBreaks_default 
  	= getAparameter<bool> (&default_pset,"noLineBreaks",false);
						// change log 5
  bool noLineBreaks 
  	= getAparameter<bool> (&dest_pset,"noLineBreaks",noLineBreaks_default);
  if (noLineBreaks) {
    dest_ctrl.setLineLength(32000);
  }
  else {
    int  lenDef = 80;
    int  lineLen_default
    	 = getAparameter<int> (&default_pset,"lineLength",lenDef);
						// change log 5
    int  lineLen = getAparameter<int> (&dest_pset,"lineLength",lineLen_default);
    if (lineLen != lenDef) {
      dest_ctrl.setLineLength(lineLen);
    }
  }

  // if indicated, suppress time stamps in this destination's output
  bool suppressTime_default 
  	= getAparameter<bool> (&default_pset,"noTimeStamps",false);
  bool suppressTime 
  	= getAparameter<bool> (&dest_pset,"noTimeStamps",suppressTime_default);
  if (suppressTime) {
    dest_ctrl.suppressTime();
  }

}  // MessageLoggerScribe::configure_dest()


void
  MessageLoggerScribe::configure_external_dests()
{
  if( ! job_pset_p )  
  {
//  extern_dests.clear();				
//  change log 12, removed by change log 13
    return;
  }

  for( std::vector<NamedDestination*>::const_iterator it = extern_dests.begin()
     ; it != extern_dests.end()
     ;  ++it
     )
  {
    ELdestination *  dest_p = (*it)->dest_p().get();
    ELdestControl  dest_ctrl = admin_p->attach( *dest_p );

    // configure the newly-attached destination:
    configure_dest( dest_ctrl, (*it)->name() );
    delete *it;  // dispose of our (copy of the) NamedDestination
  }
  extern_dests.clear();
 
}  // MessageLoggerScribe::configure_external_dests

void
  MessageLoggerScribe::parseCategories (std::string const & s,
  				        std::vector<std::string> & cats)
{
  const std::string::size_type npos = std::string::npos;
        std::string::size_type i    = 0;
  while ( i != npos ) {    
    std::string::size_type j = s.find('|',i);   
    cats.push_back (s.substr(i,j-i));
    i = j;
    while ( (i != npos) && (s[i] == '|') ) ++i; 
    // the above handles cases of || and also | at end of string
  } 
  // Note:  This algorithm assigns, as desired, one null category if it
  //        encounters an empty categories string
}

void
  MessageLoggerScribe::triggerStatisticsSummaries() {
    assert (statisticsDestControls.size() == statisticsResets.size());
    for (unsigned int i = 0; i != statisticsDestControls.size(); ++i) {
      statisticsDestControls[i].summary( );
      if (statisticsResets[i]) statisticsDestControls[i].wipe( );
    }
}

} // end of namespace service  
} // end of namespace edm  
