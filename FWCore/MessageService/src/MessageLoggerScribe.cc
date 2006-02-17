// ----------------------------------------------------------------------
//
// MessageLoggerScribe.cc
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
    // attach the current destination, keeping a control handle to it:
    ELdestControl dest_ctrl;
    String filename = *it;
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
      std::ofstream * os_p = new std::ofstream(filename.c_str());
      file_ps.push_back(os_p);
      dest_ctrl = admin_p->attach( ELoutput(*os_p) );
      stream_ps[filename] = os_p;
    }
    //(*errorlog_p)( ELinfo, "added_dest") << filename << endmsg;

    // now configure this destination:
    configure_dest(dest_ctrl, filename);

  }  // for [it = destinations.begin() to end()]

  // grab list of fwkJobReports:
  vString  fwkJobReports
     = getAparameter<vString>(job_pset_p, "fwkJobReports", empty_vString);

  // dial down the early destination if other dest's are supplied:
  if( ! fwkJobReports.empty() )
    early_dest.setThreshold(ELhighestSeverity);

  // establish each fwkJobReports destination:
  for( vString::const_iterator it = fwkJobReports.begin()
     ; it != fwkJobReports.end()
     ; ++it
     )
  {
    // attach the current destination, keeping a control handle to it:
    ELdestControl dest_ctrl;
    String filename = *it;
    std::ofstream * os_p = new std::ofstream(filename.c_str());
    file_ps.push_back(os_p);
    dest_ctrl = admin_p->attach( ELfwkJobReport(*os_p) );
    stream_ps[filename] = os_p;

    // now configure this destination:
    configure_dest(dest_ctrl, filename);

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
    PSet  stat_pset 
    	= getAparameter<PSet>(job_pset_p,statname,empty_PSet);
    String filename 
        = getAparameter<String>(&stat_pset,"output",statname);
    
    // create (if statistics file does not match any destination file name)
    // or note (if statistics file matches a destination file name) the ostream
    std::ostream * os_p;
    if ( stream_ps.find(filename) == stream_ps.end() ) {
      if ( filename == "cout" ) {
        os_p = &std::cout;
      } else if ( filename == "cerr" ) {
        os_p = &std::cerr;
      } else {
	std::ofstream * osf_p = new std::ofstream(filename.c_str());
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
    configure_dest(dest_ctrl, filename);

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
  vString  empty_vString;
  PSet     empty_PSet;
  String   empty_String;

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
     = getAparameter<String>(job_pset_p,"threshold", "INFO");

  // grab default limit/timespan common to all destinations/categories:
  PSet  default_pset
     = getAparameter<PSet>(job_pset_p,"default", empty_PSet);
  int  default_limit
    = getAparameter<int>(&default_pset,"limit", -1);
  int  default_timespan
    = getAparameter<int>(&default_pset,"timespan", -1);

  // grab all of this destination's parameters:
  PSet  dest_pset = getAparameter<PSet>(job_pset_p,filename,empty_PSet);

  // grab this destination's default limit/timespan:
  PSet  dest_default_pset
     = getAparameter<PSet>(&dest_pset,"default", empty_PSet);
  int  dest_default_limit
    = getAparameter<int>(&dest_pset,"limit", default_limit);
  int  dest_default_timespan
    = getAparameter<int>(&dest_pset,"timespan", default_timespan);

  // establish this destination's threshold:
  String dest_threshold
     = getAparameter<String>(&dest_pset,"threshold", default_threshold);
  ELseverityLevel  threshold_sev(dest_threshold);
  dest_ctrl.setThreshold(threshold_sev);

  // establish this destination's limit/timespan for each of the categories:
  for( vString::const_iterator id_it = categories.begin()
     ; id_it != categories.end()
     ; ++id_it
     )
  {
    String  msgID = *id_it;
    PSet  category_pset
       = getAparameter<PSet>(&dest_pset,msgID, empty_PSet);
    int  limit
      = getAparameter<int>(&category_pset,"limit", dest_default_limit);
    int  timespan
      = getAparameter<int>(&category_pset,"timespan", dest_default_timespan);
    if( limit    >= 0 )  dest_ctrl.setLimit(msgID, limit   );
    if( timespan >= 0 )  dest_ctrl.setTimespan(msgID, timespan);
  }  // for

  // establish this destination's limit for each severity:
  for( vString::const_iterator sev_it = severities.begin()
     ; sev_it != severities.end()
     ; ++sev_it
     )
  {
    String  sevID = *sev_it;
    ELseverityLevel  severity(sevID);
    PSet  sev_pset = getAparameter<PSet>(&dest_pset,sevID, empty_PSet);
    int  limit     = getAparameter<int>(&sev_pset,"limit", -1);
    int  timespan  = getAparameter<int>(&sev_pset,"timespan", -1);
    if( limit    >= 0 )  dest_ctrl.setLimit(severity, limit   );
    if( timespan >= 0 )  dest_ctrl.setLimit(severity, timespan);
  }  // for

  // establish this destination's linebreak policy:
  bool noLineBreaks = getAparameter<bool> (&dest_pset,"noLineBreaks",false);
  if (noLineBreaks) {
    dest_ctrl.setLineLength(32000);
  }
  else {
    int  lenDef = 80;
    int  lineLen = getAparameter<int> (&dest_pset,"lineLength",lenDef);
    if (lineLen != lenDef) {
      dest_ctrl.setLineLength(lineLen);
    }
  }

  // if indicated, suppress time stamps in this destination's output
  bool suppressTime = getAparameter<bool> (&dest_pset,"noTimeStamps",false);
  if (suppressTime) {
    dest_ctrl.suppressTime();
  }

}  // MessageLoggerScribe::configure_dest()


void
  MessageLoggerScribe::configure_external_dests()
{
  if( ! job_pset_p )  return;

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
