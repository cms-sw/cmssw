#include "DQMServices/Core/src/DQMService.h"
#include "DQMServices/Core/src/DQMRootBuffer.h"
#include "DQMServices/Core/interface/DQMNet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMScope.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <pthread.h>
#include <iostream>
#include <string>
#include <memory>

// -------------------------------------------------------------------
static pthread_mutex_t	s_mutex   = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	s_avail   = PTHREAD_COND_INITIALIZER;
static pthread_cond_t	s_done    = PTHREAD_COND_INITIALIZER;
static int		s_pending = 0;
static bool		s_run     = false;

/// Acquire lock and access to the DQM core from a thread other than
/// the "main" CMSSW processing thread, such as in extra XDAQ threads.
DQMScope::DQMScope(void)
{
  // Wait until it's safe to run.
  pthread_mutex_lock(&s_mutex);
  ++s_pending;
  while (! s_run)
    pthread_cond_wait(&s_avail, &s_mutex);

  // OK, run now.  We are holding the mutex until calling scope exits.
}

/// Release access lock to the DQM core.
DQMScope::~DQMScope(void)
{
  // Release the mutex for someone else to use the DQM core.
  if (--s_pending == 0)
    pthread_cond_signal(&s_done);
  pthread_mutex_unlock(&s_mutex);
}

/// Let other threads use the DQM.
static void
respondToOtherThreads(const edm::Event &, const edm::EventSetup &)
{
  pthread_mutex_lock(&s_mutex);
  if (s_pending > 0)
  {
    s_run = true;
    pthread_cond_broadcast(&s_avail);

    while (s_pending > 0)
      pthread_cond_wait(&s_done, &s_mutex);

    s_run = false;
  }

  pthread_mutex_unlock(&s_mutex);
}

/// Release access to the DQM core for good.
static void
releaseDQM(void)
{
  pthread_mutex_lock(&s_mutex);
  s_run = true;
  pthread_cond_broadcast(&s_avail);
  pthread_mutex_unlock(&s_mutex);
}

// -------------------------------------------------------------------
DQMService::DQMService(const edm::ParameterSet &pset, edm::ActivityRegistry &ar)
  : store_(&*edm::Service<DQMStore>()),
    net_(0),
    lastFlush_(0),
    publishFrequency_(5.0)
{
  ar.watchPostProcessEvent(&respondToOtherThreads);
  ar.watchPostProcessEvent(this, &DQMService::flush);

  ar.watchPostEndJob(this, &DQMService::shutdown);
  ar.watchPostEndJob(&releaseDQM);

  std::string host = pset.getUntrackedParameter<std::string>("collectorHost", ""); 
  int port = pset.getUntrackedParameter<int>("collectorPort", 9090);
  bool verbose = pset.getUntrackedParameter<bool>("verbose", false);
  publishFrequency_ = pset.getUntrackedParameter<double>("publishFrequency", publishFrequency_);

  if (host != "" && port > 0)
  {
    net_ = new DQMBasicNet;
    net_->debug(verbose);
    net_->updateToCollector(host, port);
    net_->start();
  }
}

DQMService::~DQMService(void)
{
  shutdown();
}

// Flush updates to the network layer at the end of each event.  This
// is the only point at which the main application and the network
// layer interact outside initialisation and exit.
void
DQMService::flush(const edm::Event &, const edm::EventSetup &)
{
  // Avoid sending updates excessively often.
  uint64_t version = lat::Time::current().ns();
  double vtime = version * 1e-9;
  if (vtime - lastFlush_ < publishFrequency_)
    return;
  lastFlush_ = vtime;

  // OK, send an update.
  if (net_)
  {
    // Lock the network layer so we can modify the data.
    net_->lock();
    bool updated = false;

    // Find updated contents and update the network cache.
    DQMStore::MEMap::iterator i, e;
    for (i = store_->data_.begin(), e = store_->data_.end(); i != e; ++i)
    {
      MonitorElement &me = i->second;
      if (! me.wasUpdated())
	continue;
      
      assert(me.data_.object);

      DQMNet::Object o;
      o.version = version;
      o.name = me.data_.name;
      o.tags = me.data_.tags;
      o.object = 0;
      o.reference = 0;
      o.flags = me.data_.flags;
      o.lastreq = 0;

      DQMRootBuffer buffer (DQMRootBuffer::kWrite);
      buffer.WriteObject(me.data_.object);
      if (me.data_.reference)
	buffer.WriteObject(me.data_.reference);
      else
	buffer.WriteObjectAny(0, 0);

      // Save the quality test results.
      DQMNet::QReports::iterator qi, qe;
      for (qi = me.data_.qreports.begin(), qe = me.data_.qreports.end(); qi != qe; ++qi)
      {
	TObjString s (me.qualityTagString(*qi).c_str());
	buffer.WriteObject(&s);
      }

      // Save this ensemble to the buffer.
      o.rawdata.resize(buffer.Length());
      memcpy(&o.rawdata[0], buffer.Buffer(), buffer.Length());
      net_->updateLocalObject(o);
      updated = true;
    }

    // Find removed contents and clear the network cache.
    std::vector<std::string>::iterator ri, re;
    for (ri = store_->removed_.begin(), re = store_->removed_.end(); ri != re; ++ri, updated = true)
      net_->removeLocalObject(*ri);

    // Unlock the network layer.
    net_->unlock();

    // Tell network to flush if we updated something.
    if (updated)
      net_->sendLocalChanges();
  }

  store_->reset();
}

// Disengage the network service.
void
DQMService::shutdown(void)
{
  // If we have a network, let it go.
  if (net_)
    net_->shutdown();
}
