#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/AbstractMLscribe.h"
#include "FWCore/MessageLogger/interface/ConfigurationHandshake.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cstring>
#include <iostream>

//////////////////////////////////////////////////////////////////////
//
// DO NOT replace the internal memcpy() calls by assignment or by
// any other form of copying unless you first understand in depth
// all of the alignment issues involved
//
//////////////////////////////////////////////////////////////////////

// Change Log
// 
// 1 - 3/9/07 mf
//	Addition of JOB command, to be used by --jobreport
// 2 - 6/19/07 mf
//	Addition of MOD command, to be used by --mode
// 3 - 7/24/07 mf
//	Addition of SHT command, to be used when no .cfg file was given
// 4 - 7/25/07 mf
//	Change of each mommand function to start with MLq, e.g. MLqLOG
// 5 - 8/7/07 mf
//	Addition of FLS command, to be used by FlushMessageLog
// 6 - 8/16/07 mf
//	Addition of GRP command, to be used by GroupLogStatistics
// 7 - 6/18/08 mf
//	Addition of JRS command, to be used by SummarizeInJobReport
// 8 - 10/24/08 mf
//	Support for singleThread
//

using namespace edm;


SingleConsumerQ  MessageLoggerQ::buf(buf_size, buf_depth);
edm::service::AbstractMLscribe * MessageLoggerQ::mlscribe_ptr = 0;// changeLog 8
bool MessageLoggerQ::singleThread = false;			  // changeLog 8

MessageLoggerQ::MessageLoggerQ()
{ }


MessageLoggerQ::~MessageLoggerQ()
{ }


MessageLoggerQ *
  MessageLoggerQ::instance()
{
  static MessageLoggerQ queue;
  return &queue;
}  // MessageLoggerQ::instance()

void
  MessageLoggerQ::setMLscribe_ptr(edm::service::AbstractMLscribe * m)
  								// changeLog 8
{
  mlscribe_ptr = m;
  singleThread = true; 
}  // MessageLoggerQ::setMLscribe_ptr(m)

void
  MessageLoggerQ::simpleCommand(OpCode opcode, void * operand)  // changeLog 8
{
  if (singleThread){ 
    mlscribe_ptr->runCommand(opcode, operand);
  }  else {
    SingleConsumerQ::ProducerBuffer b(buf);
    char * slot_p = static_cast<char *>(b.buffer()); 
    OpCode o = opcode;
    void * v = operand;
    std::memcpy(slot_p, &o, sizeof(OpCode));
    std::memcpy(slot_p+sizeof(OpCode), &v, sizeof(void *));
    b.commit(buf_size);
  }
} // simpleCommand

void
  MessageLoggerQ::handshakedCommand( 
  	OpCode opcode, 
	void * operand,
	std::string const & commandMnemonic )
{
  if (singleThread){ 
    try {
      mlscribe_ptr->runCommand(opcode, operand);
    } 
    catch(edm::Exception& ex)
    {
      ex << "\n The preceding exception was thrown in MessageLoggerScribe\n";
      ex << "and forwarded to the main thread from the Messages thread.";
      std::cerr << "exception from MessageLoggerQ::" 
                << commandMnemonic << " - exception what() is \n" 
    	        << ex.what(); 
      throw ex;
    }
  }  else {
    Place_for_passing_exception_ptr epp = 
    				new Pointer_to_new_exception_on_heap(0);
    ConfigurationHandshake h(operand,epp);
    SingleConsumerQ::ProducerBuffer b(buf);
    char * slot_p = static_cast<char *>(b.buffer());
    OpCode o = opcode;
    void * v(static_cast<void *>(&h));
    std::memcpy(slot_p+0             , &o, sizeof(OpCode));
    std::memcpy(slot_p+sizeof(OpCode), &v, sizeof(void *));
    Pointer_to_new_exception_on_heap ep;
    {
      boost::mutex::scoped_lock sl(h.m);       // get lock
      b.commit(buf_size);
      // wait for result to appear (in epp)
      h.c.wait(sl); // c.wait(sl) unlocks the scoped lock and sleeps till notified
      // ... and once the MessageLoggerScribe does h.c.notify_all() ... 
      ep = *h.epp;
      // finally, release the scoped lock by letting it go out of scope 
    }
    if ( ep ) {
      edm::Exception ex(*ep);
      delete ep;
      ex << "\n The preceding exception was thrown in MessageLoggerScribe\n";
      ex << "and forwarded to the main thread from the Messages thread.";
      std::cerr << "exception from MessageLoggerQ::" 
                << commandMnemonic << " - exception what() is \n" 
    		<< ex.what(); 
      throw ex;
    }  
  }
}  // handshakedCommand

void
  MessageLoggerQ::MLqEND()
{
  simpleCommand (END_THREAD, (void *)0); 
}  // MessageLoggerQ::END()

void
  MessageLoggerQ::MLqSHT()
{
  simpleCommand (SHUT_UP, (void *)0); 
}  // MessageLoggerQ::SHT()

void
  MessageLoggerQ::MLqLOG( ErrorObj * p )
{
  simpleCommand (LOG_A_MESSAGE, static_cast<void *>(p)); 
}  // MessageLoggerQ::LOG()


void
  MessageLoggerQ::MLqCFG( ParameterSet * p )
{
  handshakedCommand(CONFIGURE, p, "CFG" );
}  // MessageLoggerQ::CFG()

void
MessageLoggerQ::MLqEXT( service::NamedDestination* p )
{
  simpleCommand (EXTERN_DEST, static_cast<void *>(p)); 
}

void
  MessageLoggerQ::MLqSUM( )
{
  simpleCommand (SUMMARIZE, 0); 
}  // MessageLoggerQ::SUM()

void
  MessageLoggerQ::MLqJOB( std::string * j )
{
  simpleCommand (JOBREPORT, static_cast<void *>(j)); 
}  // MessageLoggerQ::JOB()

void
  MessageLoggerQ::MLqMOD( std::string * jm )
{
  simpleCommand (JOBMODE, static_cast<void *>(jm)); 
}  // MessageLoggerQ::MOD()


void
  MessageLoggerQ::MLqFLS(  )			// Change Log 5
{
  if (singleThread) return;
  // The ConfigurationHandshake, developed for synchronous CFG, contains a
  // place to convey exception information.  FLS does not need this, nor does
  // it need the parameter set, but we are reusing ConfigurationHandshake 
  // rather than reinventing the mechanism.
  handshakedCommand(FLUSH_LOG_Q, 0, "FLS" );
}  // MessageLoggerQ::FLS()

void
  MessageLoggerQ::MLqGRP( std::string * cat_p )  	// Change Log 6
{
  simpleCommand (GROUP_STATS, static_cast<void *>(cat_p)); 
}  // MessageLoggerQ::GRP()

void
  MessageLoggerQ::MLqJRS( std::map<std::string, double> * sum_p )
{
  handshakedCommand(FJR_SUMMARY, sum_p, "JRS" );
}  // MessageLoggerQ::CFG()

void
  MessageLoggerQ::consume( OpCode & opcode, void * & operand )
{
  SingleConsumerQ::ConsumerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  std::memcpy(&opcode , slot_p+0             , sizeof(OpCode));
  std::memcpy(&operand, slot_p+sizeof(OpCode), sizeof(void *));
  b.commit(buf_size);
}  // MessageLoggerQ::consume()

