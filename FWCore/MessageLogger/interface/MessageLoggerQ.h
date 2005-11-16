#ifndef FWCore_MessageLogger_MessageLoggerQ_h
#define FWCore_MessageLogger_MessageLoggerQ_h


#include "IOPool/Streamer/interface/EventBuffer.h"
#include <memory>


namespace edm
{

// --- forward declarations:
//

class ErrorObj;
class ParameterSet;


class MessageLoggerQ
{
public:
  // --- enumerate types of messages that can be enqueued:
  enum OpCode      // abbrev's used hereinafter
  { END_THREAD     // END
  , LOG_A_MESSAGE  // LOG
  , CONFIGURE      // CFG
  };  // OpCode

  // ---  birth via a surrogate:
  static  MessageLoggerQ *  instance();

  // ---  post a message to the queue:
  static  void  END();
  static  void  LOG( ErrorObj * p );
  static  void  CFG( ParameterSet * p );

  // ---  obtain a message from the queue:
  static  void  consume( OpCode & opcode, void * & operand );

private:
  // ---  traditional birth/death, but disallowed to users:
  MessageLoggerQ();
  ~MessageLoggerQ();

  // --- no copying:
  MessageLoggerQ( MessageLoggerQ const & );
  void  operator= ( MessageLoggerQ const & );

  // --- parameters:
  static  const int  buf_size  = sizeof(OpCode) + sizeof(void *);
  static  const int  buf_depth = 500;

  // --- data:
  static  EventBuffer  buf;

};  // MessageLoggerQ


}  // namespace edm


#endif  // FWCore_MessageLogger_MessageLoggerQ_h



