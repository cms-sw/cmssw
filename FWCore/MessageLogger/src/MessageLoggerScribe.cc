#include "FWCore/MessageLogger/interface/MessageLoggerScribe.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/ELoutput.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"

#include <cassert>
#include <iostream>
#include <fstream>


using namespace edm;


MessageLoggerScribe::MessageLoggerScribe()
: admin_p   ( ELadministrator::instance() )
, early_dest( admin_p->attach(ELoutput(std::cout)) )  // cerr later
, errorlog_p( new ErrorLog() )
{ }


MessageLoggerScribe::~MessageLoggerScribe()
{
  delete errorlog_p;
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
        (*errorlog_p)(*errorobj_p);  // route the message text
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
  // temporary:
  std::ofstream out("testfile.txt");
  admin_p->attach( ELoutput(out) );
  (*errorlog_p)( ELwarning, "configure_errorlog")
    << "warning, warning, Will Robinson!"
    << endmsg;

  //////////////////////////////////////////////////////////////////////
  // for temporary reference:
  //   ELdestControl      early_dest;
  //   ELadministrator *  admin_p;
  //   ErrorLog        *  errorlog_p;
  //////////////////////////////////////////////////////////////////////

}  // MessageLoggerScribe::configure()
