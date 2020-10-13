#ifndef FWCore_MessageLogger_MessageLoggerQ_h
#define FWCore_MessageLogger_MessageLoggerQ_h

#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <memory>

#include <string>
#include <map>
#include <set>

namespace edm {

  // --- forward declarations:
  class ErrorObj;
  class ParameterSet;
  class ELdestination;
  namespace service {
    class AbstractMLscribe;
  }

  class MessageLoggerQ {
  public:
    MessageLoggerQ(MessageLoggerQ const&) = delete;
    void operator=(MessageLoggerQ const&) = delete;

    // --- enumerate types of messages that can be enqueued:
    enum OpCode   // abbrev's used hereinafter
    { END_THREAD  // END
      ,
      LOG_A_MESSAGE  // LOG
      ,
      CONFIGURE  // CFG -- handshaked
      ,
      EXTERN_DEST  // EXT
      ,
      SUMMARIZE  // SUM
      ,
      JOBMODE  // MOD
      ,
      SHUT_UP  // SHT
      ,
      FLUSH_LOG_Q  // FLS -- handshaked
      ,
      GROUP_STATS  // GRP
      ,
      FJR_SUMMARY  // JRS -- handshaked
    };             // OpCode

    // ---  birth via a surrogate:
    static MessageLoggerQ* instance();

    // ---  post a message to the queue:
    static void MLqEND();
    static void MLqLOG(ErrorObj* p);
    static void MLqCFG(ParameterSet* p);
    static void MLqSUM();
    static void MLqMOD(std::string* jm);
    static void MLqSHT();
    static void MLqFLS();
    static void MLqGRP(std::string* cat_p);
    static void MLqJRS(std::map<std::string, double>* sum_p);

    // ---  bookkeeping for single-thread mode
    static void setMLscribe_ptr(std::shared_ptr<edm::service::AbstractMLscribe> m);

    // ---  helper for scribes
    static bool handshaked(const OpCode& op);

    // --- special control of standAlone logging behavior
    static void standAloneThreshold(edm::ELseverityLevel const& severity);
    static void squelch(std::string const& category);
    static bool ignore(edm::ELseverityLevel const& severity, std::string const& category);

  private:
    // ---  traditional birth/death, but disallowed to users:
    MessageLoggerQ();
    ~MessageLoggerQ();

    // ---  place an item onto the queue, or execute the command directly
    static void simpleCommand(OpCode opcode, void* operand);
    static void handshakedCommand(OpCode opcode, void* operand, std::string const& commandMnemonic);

    // --- data:
    CMS_THREAD_SAFE static std::shared_ptr<edm::service::AbstractMLscribe> mlscribe_ptr;
    CMS_THREAD_SAFE static edm::ELseverityLevel threshold;
    CMS_THREAD_SAFE static std::set<std::string> squelchSet;

  };  // MessageLoggerQ

}  // namespace edm

#endif  // FWCore_MessageLogger_MessageLoggerQ_h
