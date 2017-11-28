#ifndef FWCore_MessageService_MainThreadMLscribe_h
#define FWCore_MessageService_MainThreadMLscribe_h

#include "FWCore/MessageLogger/interface/AbstractMLscribe.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// I believe the below are not needed:

#include <memory>

#include <iosfwd>
#include <vector>
#include <map>

#include <iostream>

namespace edm {
  namespace service {

    // ----------------------------------------------------------------------
    //
    // MainThreadMLscribe.h
    //
    // This class is a concrete of AbstractMessageLoggerScribe
    // Its purpose exists ONLY if there is a second thread running the workhorse
    // scrribe.  In that case, the workhorse will be consuming from a
    // SingleConsumerQ, and this class is the one that places the item on said
    // queue.  It does work that used to be the realm of MessageLoggerQ.
    //
    // Changes:
    //
    // 0 - 8/7/09  	Initial version mf and crj
    //
    // -----------------------------------------------------------------------

    class ThreadQueue;

    class MainThreadMLscribe : public AbstractMLscribe {
    public:
      // ---  birth/death:
      MainThreadMLscribe(std::shared_ptr<ThreadQueue> tqp);
      ~MainThreadMLscribe() override;

      // --- receive and act on messages:

      void runCommand(MessageLoggerQ::OpCode opcode, void* operand) override;

    private:
      edm::propagate_const<std::shared_ptr<ThreadQueue>> m_queue;
    };  // MainThreadMLscribe

  }  // end of namespace service
}  // namespace edm

#endif  // FWCore_MessageService_MainThreadMLscribe_h
