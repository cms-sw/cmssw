#ifndef FWCore_MessageLogger_MessageLoggerQ_h
#define FWCore_MessageLogger_MessageLoggerQ_h

#include "FWCore/Utilities/interface/SingleConsumerQ.h"


namespace edm
{

// --- forward declarations:
class ErrorObj;
class ParameterSet;
class ELdestination;
namespace service {
class NamedDestination;
}


class MessageLoggerQ
{
public:
  // --- enumerate types of messages that can be enqueued:
  enum OpCode      // abbrev's used hereinafter
  { END_THREAD     // END
  , LOG_A_MESSAGE  // LOG
  , CONFIGURE      // CFG
  , EXTERN_DEST    // EXT
  , SUMMARIZE      // SUM
  , JOBREPORT      // JOB
  , JOBMODE        // MOD
  , SHUT_UP        // SHT
  };  // OpCode

  // ---  birth via a surrogate:
  static  MessageLoggerQ *  instance();

  // ---  post a message to the queue:
  static  void  END();
  static  void  LOG( ErrorObj * p );
  static  void  CFG( ParameterSet * p );
  static  void  EXT( service::NamedDestination* p );
  static  void  SUM();
  static  void  JOB( std::string * j );
  static  void  MOD( std::string * jm );
  static  void  SHT();

  // ---  obtain a message from the queue:
  static  void  consume( OpCode & opcode, void * & operand );

private:
  // ---  traditional birth/death, but disallowed to users:
  MessageLoggerQ();
  ~MessageLoggerQ();

  // --- no copying:
  MessageLoggerQ( MessageLoggerQ const & );
  void  operator = ( MessageLoggerQ const & );

  // --- buffer parameters:
  static  const int  buf_depth = 500;
  static  const int  buf_size  = sizeof(OpCode)
                               + sizeof(void *);

  // --- data:
  static  SingleConsumerQ  buf;

};  // MessageLoggerQ


}  // namespace edm


#endif  // FWCore_MessageLogger_MessageLoggerQ_h
