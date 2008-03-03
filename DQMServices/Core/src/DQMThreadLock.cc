#include "DQMServices/Core/interface/DQMThreadLock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <pthread.h>
#define NO_THREAD ((pthread_t) -1)

static bool		s_interlocking = false;
static pthread_mutex_t	s_dqmmutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	s_dqmavail = PTHREAD_COND_INITIALIZER;
static pthread_t	s_lockholder = NO_THREAD;
static int		s_busy = 0;

// -------------------------------------------------------------------
/// Lock DQM core in the CMSSW thread.
static void
getCMSSWLock(void)
{
  // Lock and mark the DQM core busy.
  if (s_interlocking)
  {
    pthread_mutex_lock(&s_dqmmutex);
    assert((s_busy == 0 && s_lockholder == NO_THREAD)
	   || (s_busy > 0 && s_lockholder == pthread_self()));
    if (++s_busy == 1)
      s_lockholder = pthread_self();
    pthread_mutex_unlock(&s_dqmmutex);
  }
}

/// Release DQM core in the CMSSW thread.
static void
releaseCMSSWLock(void)
{
  // Lock and mark the DQM core less busy.  If this was the last use
  // of DQM core in this thread, signal the condition variable so
  // other threads get a chance to use the core.
  if (s_interlocking)
  {
    pthread_mutex_lock(&s_dqmmutex);
    assert(s_lockholder == pthread_self());
    assert(s_busy > 0);
    if (--s_busy == 0)
    {
      s_lockholder = NO_THREAD;
      pthread_cond_signal(&s_dqmavail);
    }
    pthread_mutex_unlock(&s_dqmmutex);
  }
}


// -------------------------------------------------------------------
/// Unlock DQM core at the end of (CMSSW) job initialisation.
static void
unlockAtStart(void)
{
  // Release the busy mark on the DQM core, ensuring we are actually
  // holding the lock as per job initialisation settings.
  if (s_interlocking)
  {
    pthread_mutex_lock(&s_dqmmutex);
    assert(s_lockholder == pthread_self());
    assert(s_busy == 1);
    s_busy = 0;
    s_lockholder = NO_THREAD;
    pthread_cond_signal(&s_dqmavail);
    pthread_mutex_unlock(&s_dqmmutex);
  }
}

/// Lock DQM core before event processing begins in the CMSSW thread.
static void
lockForEvent(const edm::EventID &, const edm::Timestamp &)
{ getCMSSWLock(); }

/// Unlock DQM core after event processing has ended.
static void
unlockAfterEvent(const edm::Event &, const edm::EventSetup &)
{ releaseCMSSWLock(); }

/// Unlock DQM core at the end of the (CMSSW) processing.
static void
unlockAtEnd(void)
{
  // Release any locks on the DQM core.  Be graceful here in case
  // exceptions etc. left the core in the locked state.
  if (s_interlocking)
  {
    pthread_mutex_lock(&s_dqmmutex);
    s_busy = 0;
    s_lockholder = NO_THREAD;
    pthread_cond_signal(&s_dqmavail);
    pthread_mutex_unlock(&s_dqmmutex);
  }
}

// -------------------------------------------------------------------
/// Acquire lock and access to the DQM core from a thread other than
/// the "main" CMSSW processing thread, such as in extra XDAQ threads.
DQMThreadLock::ExtraThread::ExtraThread(void)
{
  // Make sure interlocking was enabled.
  if (! s_interlocking)
    throw cms::Exception("DQMThreadLock")
      << "DQMThreadLock::ExtraThreadLock used but DQM access to other threads is disabled."
      << " Please add 'replace DQMThreadLock.enabled = true' to your .cfg file.";

  // Lock.
  pthread_mutex_lock(&s_dqmmutex);
  while (s_busy > 0)
    pthread_cond_wait(&s_dqmavail, &s_dqmmutex);

  assert(s_lockholder == NO_THREAD);
  s_lockholder = pthread_self();
}

/// Release access lock to the DQM core.
DQMThreadLock::ExtraThread::~ExtraThread(void)
{
  // Release the lock.
  if (s_interlocking)
  {
    assert(s_busy == 0);
    assert(s_lockholder == pthread_self());
    s_lockholder = NO_THREAD;
    pthread_mutex_unlock(&s_dqmmutex);
  }
}

// -------------------------------------------------------------------
/// Acquire lock and access to the DQM core in an EDM service in the
/// "main" CMSSW processing thread.  Any access to the DQM core by a a
/// service must be protected as the DQMThreadLock service itself may
/// release the lock before the other services.
DQMThreadLock::EDMService::EDMService(void)
{ getCMSSWLock(); }

/// Release access lock to the DQM core.
DQMThreadLock::EDMService::~EDMService(void)
{ releaseCMSSWLock(); }

// -------------------------------------------------------------------
/// Initialise DQM interlocking based on job configuration.
DQMThreadLock::DQMThreadLock(const edm::ParameterSet &pset, edm::ActivityRegistry &ar)
{
  // Check if thread locking is needed at all.
  if (! (s_interlocking = pset.getUntrackedParameter<bool>("enabled", false)))
    return;

  // Lock the core from the start up.  We'll release after initialisation.
  pthread_mutex_lock(&s_dqmmutex);
  assert(s_busy == 0);
  assert(s_lockholder == NO_THREAD);
  s_lockholder = pthread_self();
  s_busy = 1;
  pthread_mutex_unlock(&s_dqmmutex);

  // Install lock management hooks.
  ar.watchPostBeginJob(&unlockAtStart);
  ar.watchPreProcessEvent(&lockForEvent);
  ar.watchPostProcessEvent(&unlockAfterEvent);
  ar.watchPostEndJob(&unlockAtEnd);
  ar.watchJobFailure(&unlockAtEnd);
}

DQMThreadLock::~DQMThreadLock(void)
{
  assert(s_busy == 0);
  assert(s_lockholder == NO_THREAD);
}
