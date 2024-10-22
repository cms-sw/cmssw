
#include "Fireworks/Core/interface/TRootXTReq.h"

#include "TCondition.h"
#include "TThread.h"
#include "TMutex.h"
#include <TSysEvtHandler.h>
#include <TSystem.h>
#include <TTimer.h>

// Bloody root threads do not provide signal delivery.
#include <csignal>

// TRootXTReq

//______________________________________________________________________________
//
// Abstract base-class for delivering cross-thread requests into ROOT
// main thread.
// Sub-classes must implement the Act() method.
// Two methods are available to request execution:
//
// - ShootRequest()
//     Request execution and return immediately.
//     The request object is deleted in the main thread.
//
// - ShootRequestAndWait()
//   Request execution and wait until execution is finished.
//   This way results can be returned to the caller in data-members.
//   It is callers responsibility to delete the request object.
//   It can also be reused.
//
//
// Global queue and locks are implemented via static members of this
// class.
//
// Must be initialized from the main thread like this:
//   TRootXTReq::Bootstrap(TThread::SelfId());

TRootXTReq::lpXTReq_t TRootXTReq::sQueue;
pthread_t TRootXTReq::sRootThread = 0;
TMutex *TRootXTReq::sQueueMutex = nullptr;
TSignalHandler *TRootXTReq::sSigHandler = nullptr;
bool TRootXTReq::sSheduled = false;

//==============================================================================

TRootXTReq::TRootXTReq(const char *n) : m_return_condition(nullptr), mName(n) {}

TRootXTReq::~TRootXTReq() { delete m_return_condition; }

//------------------------------------------------------------------------------

void TRootXTReq::post_request() {
  TLockGuard _lck(sQueueMutex);

  sQueue.push_back(this);

  if (!sSheduled) {
    sSheduled = true;
    pthread_kill(sRootThread, SIGUSR1);
  }
}

void TRootXTReq::ShootRequest() {
  // Places request into the queue and requests execution in Rint thread.
  // It returns immediately after that, without waiting for execution.
  // The request is deleted after execution.

  if (m_return_condition) {
    delete m_return_condition;
    m_return_condition = nullptr;
  }

  post_request();
}

void TRootXTReq::ShootRequestAndWait() {
  // Places request into the queue, requests execution in Rint thread and
  // waits for the execution to be completed.
  // The request is not deleted after execution as it might carry return
  // value.
  // The same request can be reused several times.

  if (!m_return_condition)
    m_return_condition = new TCondition;

  m_return_condition->GetMutex()->Lock();

  post_request();

  m_return_condition->Wait();
  m_return_condition->GetMutex()->UnLock();
}

//==============================================================================

class RootSig2XTReqHandler : public TSignalHandler {
private:
  class XTReqTimer : public TTimer {
  public:
    XTReqTimer() : TTimer() {}
    ~XTReqTimer() override {}

    void FireAway() {
      Reset();
      gSystem->AddTimer(this);
    }

    Bool_t Notify() override {
      gSystem->RemoveTimer(this);
      TRootXTReq::ProcessQueue();
      return kTRUE;
    }
  };

  XTReqTimer mTimer;

public:
  RootSig2XTReqHandler() : TSignalHandler(kSigUser1), mTimer() { Add(); }
  ~RootSig2XTReqHandler() override {}

  Bool_t Notify() override {
    printf("Usr1 Woof Woof in Root thread! Starting Timer.\n");
    mTimer.FireAway();
    return kTRUE;
  }
};

//------------------------------------------------------------------------------

void TRootXTReq::Bootstrap(pthread_t root_thread) {
  static const TString _eh("TRootXTReq::Bootstrap ");

  if (sRootThread != 0)
    throw _eh + "Already initialized.";

  sRootThread = root_thread;
  sQueueMutex = new TMutex(kTRUE);
  sSigHandler = new RootSig2XTReqHandler;
}

void TRootXTReq::Shutdown() {
  static const TString _eh("TRootXTReq::Shutdown ");

  if (sRootThread == 0)
    throw _eh + "Have not beem initialized.";

  // Should lock and drain queue ... or sth.

  sRootThread = 0;
  delete sSigHandler;
  sSigHandler = nullptr;
  delete sQueueMutex;
  sQueueMutex = nullptr;
}

void TRootXTReq::ProcessQueue() {
  printf("Timer fired, processing queue.\n");

  while (true) {
    TRootXTReq *req = nullptr;
    {
      TLockGuard _lck(sQueueMutex);

      if (!sQueue.empty()) {
        req = sQueue.front();
        sQueue.pop_front();
      } else {
        sSheduled = false;
        break;
      }
    }

    req->Act();

    if (req->m_return_condition) {
      req->m_return_condition->GetMutex()->Lock();
      req->m_return_condition->Signal();
      req->m_return_condition->GetMutex()->UnLock();
    } else {
      delete req;
    }
  }
}
