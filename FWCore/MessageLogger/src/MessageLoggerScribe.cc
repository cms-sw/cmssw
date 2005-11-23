#include "FWCore/MessageLogger/interface/MessageLoggerScribe.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/ELoutput.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


using namespace edm;


MessageLoggerScribe::MessageLoggerScribe()
: admin_p   ( ELadministrator::instance() )
, early_dest( admin_p->attach(ELoutput(std::cerr, false)) )
, errorlog_p( new ErrorLog() )
, file_ps   ( )
{ }


MessageLoggerScribe::~MessageLoggerScribe()
{
  delete errorlog_p;
  for( ;  not file_ps.empty();  file_ps.pop_back() )  {
    delete file_ps.back();
  }
}


void
  MessageLoggerScribe::run()
{
  MessageLoggerQ::OpCode  opcode;
  void *                  operand;

  bool  done = false;
  do  {
    MessageLoggerQ::consume(opcode, operand);
    switch(opcode)  {
      default:  {
        assert(false);
	break;
      }
      case MessageLoggerQ::END_THREAD:  {
        assert( operand == 0 );
        done = true;
        break;
      }
      case MessageLoggerQ::LOG_A_MESSAGE:  {
        ErrorObj *  errorobj_p = static_cast<ErrorObj *>(operand);
	//std::cout << "MessageLoggerQ::LOG_A_MESSAGE " << errorobj_p << '\n';
        (*errorlog_p)( *errorobj_p );  // route the message text
        delete errorobj_p;  // dispose of the message text
        break;
      }
      case MessageLoggerQ::CONFIGURE:  {
        ParameterSet *  pset_p = static_cast<ParameterSet *>(operand);
        configure_errorlog( pset_p );
        delete pset_p;  // dispose of our (copy of the) ParameterSet
        break;
      }
    }  // switch

  } while(! done);

}  // MessageLoggerScribe::run()

void
  MessageLoggerScribe::configure_errorlog( ParameterSet const * p )
{
  typedef std::string          String;
  typedef std::vector<String>  vString;
  typedef ParameterSet         PSet;

  vString      empty_vString;
  PSet         empty_PSet;
  std::string  empty_String;

  // no longer need default destination:
  early_dest.setThreshold(ELhighestSeverity);

  vString  filenames
     = p->getUntrackedParameter<vString>("files", empty_vString);
  for( vString::const_iterator it = filenames.begin()
     ; it != filenames.end()
     ; ++it
     )
  {
    ELdestControl dest_ctrl;
    std::string filename = *it;
    if( filename == "cout" )  {
      dest_ctrl = admin_p->attach( ELoutput(std::cout) );
    }
    else if( filename == "cerr" )  {
      early_dest.setThreshold(ELzeroSeverity);  // or ELerror?
      dest_ctrl = early_dest;
    }
    else  {
      std::ofstream * os_p = new std::ofstream(filename.c_str());
      file_ps.push_back(os_p);
      dest_ctrl = admin_p->attach( ELoutput(*os_p) );
    }
    //(*errorlog_p)( ELinfo, "added_dest") << filename << endmsg;

    PSet pset = p->getUntrackedParameter<PSet>(filename,empty_PSet);

    // Threshold processing:
    {
      String severity_name
        = pset.getUntrackedParameter<String>("setThreshold", empty_String);
      ELseverityLevel lev = ELseverityLevel(severity_name);
      if( lev != ELunspecified )
        dest_ctrl.setThreshold(lev);
    }

    // Limit processing:
    {
      PSet limit_info
        = pset.getUntrackedParameter<PSet>("setLimit", empty_PSet);
      int limit
        = limit_info.getUntrackedParameter<int>("limit", -1);
      if( limit >= 0 )  {
        String severity_name
          = limit_info.getUntrackedParameter<String>("severity", empty_String);
        ELseverityLevel severity = ELseverityLevel(severity_name);
        if( severity != ELunspecified )
          dest_ctrl.setLimit(severity, limit);
        String msgID
          = limit_info.getUntrackedParameter<String>("messageID", "*");
        dest_ctrl.setLimit(msgID, limit);
       }
     }

    // Timespan processing:
    {
      PSet timespan_info
        = pset.getUntrackedParameter<PSet>("setTimespan", empty_PSet);
      int seconds
        = timespan_info.getUntrackedParameter<int>("seconds", -1);
      if( seconds >= 0 )  {
        String severity_name
          = timespan_info.getUntrackedParameter<String>("severity", empty_String);
        ELseverityLevel severity = ELseverityLevel(severity_name);
        if( severity != ELunspecified )
          dest_ctrl.setLimit(severity, seconds);
        String msgID
          = timespan_info.getUntrackedParameter<String>("messageID", "*");
        dest_ctrl.setLimit(msgID, seconds);
      }
    }

  }  // for


  //////////////////////////////////////////////////////////////////////
  // for temporary reference:
  //   ELdestControl                early_dest;
  //   ELadministrator           *  admin_p;
  //   ErrorLog                  *  errorlog_p;
  //   std::vector<std::ofstream *> file_ps;
  //////////////////////////////////////////////////////////////////////

}  // MessageLoggerScribe::configure()
