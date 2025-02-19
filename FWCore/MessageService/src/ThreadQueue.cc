// ----------------------------------------------------------------------
//
// ThreadQueue.cc
//
// Changes:
//
// 

#include "FWCore/MessageService/interface/ThreadQueue.h"
#include <cstring>

namespace edm {
namespace service {

ThreadQueue::ThreadQueue() 
  : m_buf (buf_size, buf_depth)
{
}
ThreadQueue::~ThreadQueue() {}

void  
ThreadQueue::
produce(MessageLoggerQ::OpCode  o, void * operand)
{
  SingleConsumerQ::ProducerBuffer b(m_buf);
  char * slot_p = static_cast<char *>(b.buffer());
  void * v = operand;
  std::memcpy(slot_p+0             , &o, sizeof(MessageLoggerQ::OpCode));
  std::memcpy(slot_p+sizeof(MessageLoggerQ::OpCode), &v, sizeof(void *));
  b.commit(buf_size);
} // runCommand
  
void  
ThreadQueue::
consume( MessageLoggerQ::OpCode & opcode, void * & operand )
{
  SingleConsumerQ::ConsumerBuffer b(m_buf);  // Look -- CONSUMER buffer
  char * slot_p = static_cast<char *>(b.buffer());
  std::memcpy(&opcode , slot_p+0             , sizeof(MessageLoggerQ::OpCode));
  std::memcpy(&operand, slot_p+sizeof(MessageLoggerQ::OpCode), sizeof(void *));
  b.commit(buf_size);
}


  

} // end of namespace service  
} // end of namespace edm  
