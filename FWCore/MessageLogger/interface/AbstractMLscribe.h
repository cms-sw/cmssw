#ifndef FWCore_MessageLogger_AbstractMLscribe_h
#define FWCore_MessageLogger_AbstractMLscribe_h

#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"

namespace edm {
  namespace service {

    class AbstractMLscribe {
    public:
      // ---  birth/death:
      AbstractMLscribe();

      // --- no copying:
      AbstractMLscribe(AbstractMLscribe const &) = delete;
      void operator=(AbstractMLscribe const &) = delete;

      virtual ~AbstractMLscribe();

      // ---  methods needed for logging
      virtual void runCommand(MessageLoggerQ::OpCode opcode, void *operand);

    };  // AbstractMLscribe

  }  // end of namespace service
}  // namespace edm

#endif  // FWCore_MessageLogger_AbstractMLscribe_h
