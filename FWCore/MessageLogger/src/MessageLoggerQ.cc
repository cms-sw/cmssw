#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
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

using namespace edm;


SingleConsumerQ  MessageLoggerQ::buf(buf_size, buf_depth);


MessageLoggerQ::MessageLoggerQ()
{ }


MessageLoggerQ::~MessageLoggerQ()
{ }


MessageLoggerQ *
  MessageLoggerQ::instance()
{
  static MessageLoggerQ *  instance = new MessageLoggerQ();
  return instance;
}  // MessageLoggerQ::instance()

void
  MessageLoggerQ::MLqEND()
{
  SingleConsumerQ::ProducerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  OpCode o(END_THREAD);
  void * v(0);

  std::memcpy(slot_p               , &o, sizeof(OpCode));
  std::memcpy(slot_p+sizeof(OpCode), &v, sizeof(void *));
  b.commit(buf_size);
}  // MessageLoggerQ::END()

void
  MessageLoggerQ::MLqSHT()
{
  SingleConsumerQ::ProducerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  OpCode o(SHUT_UP);
  void * v(0);

  std::memcpy(slot_p               , &o, sizeof(OpCode));
  std::memcpy(slot_p+sizeof(OpCode), &v, sizeof(void *));
  b.commit(buf_size);
}  // MessageLoggerQ::SHT()

void
  MessageLoggerQ::MLqLOG( ErrorObj * p )
{
  SingleConsumerQ::ProducerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  OpCode o(LOG_A_MESSAGE);
  void * v(static_cast<void *>(p));

  std::memcpy(slot_p+0             , &o, sizeof(OpCode));
  std::memcpy(slot_p+sizeof(OpCode), &v, sizeof(void *));
  b.commit(buf_size);
}  // MessageLoggerQ::LOG()


void
  MessageLoggerQ::MLqCFG( ParameterSet * p )
{
  Place_for_passing_exception_ptr epp = new Pointer_to_new_exception_on_heap(0);
  ConfigurationHandshake h((void*)p,epp);
  SingleConsumerQ::ProducerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  OpCode o(CONFIGURE);
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
    std::cerr << "exception from MessageLoggerQ::CFG - exception what() is \n" 
    		<< ex.what(); 
    throw ex;
  }  
}  // MessageLoggerQ::CFG()

void
MessageLoggerQ::MLqEXT( service::NamedDestination* p )
{
  SingleConsumerQ::ProducerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  OpCode o(EXTERN_DEST);
  void * v(static_cast<void *>(p));

  std::memcpy(slot_p+0             , &o, sizeof(OpCode));
  std::memcpy(slot_p+sizeof(OpCode), &v, sizeof(void *));
  b.commit(buf_size);  
}

void
  MessageLoggerQ::MLqSUM( )
{
  SingleConsumerQ::ProducerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  OpCode o(SUMMARIZE);
  void * v(0);

  std::memcpy(slot_p               , &o, sizeof(OpCode));
  std::memcpy(slot_p+sizeof(OpCode), &v, sizeof(void *));
  b.commit(buf_size);
}  // MessageLoggerQ::SUM()

void
  MessageLoggerQ::MLqJOB( std::string * j )
{
  SingleConsumerQ::ProducerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  OpCode o(JOBREPORT);
  void * v(static_cast<void *>(j));

  std::memcpy(slot_p+0             , &o, sizeof(OpCode));
  std::memcpy(slot_p+sizeof(OpCode), &v, sizeof(void *));
  b.commit(buf_size);
}  // MessageLoggerQ::JOB()

void
  MessageLoggerQ::MLqMOD( std::string * jm )
{
  SingleConsumerQ::ProducerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  OpCode o(JOBMODE);
  void * v(static_cast<void *>(jm));

  std::memcpy(slot_p+0             , &o, sizeof(OpCode));
  std::memcpy(slot_p+sizeof(OpCode), &v, sizeof(void *));
  b.commit(buf_size);
}  // MessageLoggerQ::MOD()


void
  MessageLoggerQ::consume( OpCode & opcode, void * & operand )
{
  SingleConsumerQ::ConsumerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  std::memcpy(&opcode , slot_p+0             , sizeof(OpCode));
  std::memcpy(&operand, slot_p+sizeof(OpCode), sizeof(void *));
  b.commit(buf_size);
}  // MessageLoggerQ::consume()

void
  MessageLoggerQ::MLqFLS(  )			// Change Log 5
{
  // The ConfigurationHandshake, developed for synchronous CFG, contains a
  // place to convey exception information.  FLS does not need this, nor does
  // it need the parameter set, but we are reusing ConfigurationHandshake 
  // rather than reinventing the mechanism.
  Place_for_passing_exception_ptr epp = new Pointer_to_new_exception_on_heap(0);
  ParameterSet * p = 0;
  ConfigurationHandshake h(p,epp);
  SingleConsumerQ::ProducerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  OpCode o(FLUSH_LOG_Q);
  void * v(static_cast<void *>(&h));

  std::memcpy(slot_p+0             , &o, sizeof(OpCode));
  std::memcpy(slot_p+sizeof(OpCode), &v, sizeof(void *));
  {
    boost::mutex::scoped_lock sl(h.m);       // get lock
    b.commit(buf_size);
    // wait for result to appear (in epp)
    h.c.wait(sl); // c.wait(sl) unlocks the scoped lock and sleeps till notified
    // ... and once the MessageLoggerScribe does h.c.notify_all() ... 
    // finally, release the scoped lock by letting it go out of scope 
  }
}  // MessageLoggerQ::FLS()

void
  MessageLoggerQ::MLqGRP( std::string * cat_p )  	// Change Log 6
{
  SingleConsumerQ::ProducerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  OpCode o(GROUP_STATS);
  void * v(static_cast<void *>(cat_p));

  std::memcpy(slot_p+0             , &o, sizeof(OpCode));
  std::memcpy(slot_p+sizeof(OpCode), &v, sizeof(void *));
  b.commit(buf_size);
}  // MessageLoggerQ::GRP()

void
  MessageLoggerQ::MLqJRS( std::map<std::string, double> * sum_p )
{
  Place_for_passing_exception_ptr epp = new Pointer_to_new_exception_on_heap(0);
  ConfigurationHandshake h((void*)sum_p,epp);
  SingleConsumerQ::ProducerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  OpCode o(FJR_SUMMARY);
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
    ex << "and forwarded to the main thread from the Messages thread.\n";
    std::cerr << "exception from MessageLoggerQ::JRS - exception what() is \n" 
    		<< ex.what(); 
    throw ex;
  }  
}  // MessageLoggerQ::CFG()

