#ifndef FWCore_MessageLogger_MessageLoggerQ_h
#define FWCore_MessageLogger_MessageLoggerQ_h

#include "FWCore/Utilities/interface/SingleConsumerQ.h"

#include <map>

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
  , FLUSH_LOG_Q    // FLS
  , GROUP_STATS    // GRP
  , FJR_SUMMARY    // JRS
  };  // OpCode

  // ---  birth via a surrogate:
  static  MessageLoggerQ *  instance();

  // ---  post a message to the queue:
  static  void  MLqEND();
  static  void  MLqLOG( ErrorObj * p );
  static  void  MLqCFG( ParameterSet * p );
  static  void  MLqEXT( service::NamedDestination* p );
  static  void  MLqSUM();
  static  void  MLqJOB( std::string * j );
  static  void  MLqMOD( std::string * jm );
  static  void  MLqSHT();
  static  void  MLqFLS();
  static  void  MLqGRP(std::string * cat_p);
  static  void  MLqJRS(std::map<std::string, double> * sum_p);

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
