#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include <cstring>


//////////////////////////////////////////////////////////////////////
//
// DO NOT replace the internal memcpy() calls by assignment or by
// any other form of copying unless you first understand in depth
// all of the alignment issues involved
//
//////////////////////////////////////////////////////////////////////


namespace edm
{


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
  MessageLoggerQ::END()
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
  MessageLoggerQ::LOG( ErrorObj * p )
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
  MessageLoggerQ::CFG( ParameterSet * p )
{
  SingleConsumerQ::ProducerBuffer b(buf);
  char * slot_p = static_cast<char *>(b.buffer());

  OpCode o(CONFIGURE);
  void * v(static_cast<void *>(p));

  std::memcpy(slot_p+0             , &o, sizeof(OpCode));
  std::memcpy(slot_p+sizeof(OpCode), &v, sizeof(void *));
  b.commit(buf_size);
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


}  // namespace edm
