#if !WITHOUT_CMS_FRAMEWORK
# include "DQMServices/Core/src/DQMService.h"
# include "DQMServices/Core/interface/DQMNet.h"
# include "DQMServices/Core/interface/DQMStore.h"
# include "DQMServices/Core/interface/DQMScope.h"
# include "DQMServices/Core/interface/MonitorElement.h"
# include "FWCore/ServiceRegistry/interface/Service.h"
# include "classlib/utils/Regexp.h"
# include "classlib/utils/Error.h"
# include <mutex>
# include <iostream>
# include <string>
# include <memory>
#include "TBufferFile.h"

// -------------------------------------------------------------------
static std::recursive_mutex s_mutex;

/// Acquire lock and access to the DQM core from a thread other than
/// the "main" CMSSW processing thread, such as in extra XDAQ threads.
DQMScope::DQMScope(void)
{ s_mutex.lock(); }

/// Release access lock to the DQM core.
DQMScope::~DQMScope(void)
{ s_mutex.unlock(); }

/// Restrict access to the DQM core.
static void
restrictDQMAccess(void)
{ s_mutex.lock(); }

static void
restrictDQMAccessM(const edm::ModuleDescription &)
{ restrictDQMAccess(); }

/// Release access to the DQM core.
static void
releaseDQMAccess(void)
{ s_mutex.unlock(); }

static void
releaseDQMAccessM(const edm::ModuleDescription &)
{ releaseDQMAccess(); }

// -------------------------------------------------------------------
DQMService::DQMService(const edm::ParameterSet &pset, edm::ActivityRegistry &ar)
  : store_(&*edm::Service<DQMStore>()),
    net_(0),
    filter_(0),
    lastFlush_(0),
    publishFrequency_(5.0)
{
  ar.watchPreSourceConstruction(&restrictDQMAccessM);
  ar.watchPostSourceConstruction(&releaseDQMAccessM);
  ar.watchPreSource(&restrictDQMAccess);
  ar.watchPostSource(&releaseDQMAccess);
  ar.watchPreModule(&restrictDQMAccessM);
  ar.watchPostModule(&releaseDQMAccessM);
  ar.watchPostProcessEvent(this, &DQMService::flush);
  ar.watchPostEndJob(this, &DQMService::shutdown);

  std::string host = pset.getUntrackedParameter<std::string>("collectorHost", ""); 
  int port = pset.getUntrackedParameter<int>("collectorPort", 9090);
  bool verbose = pset.getUntrackedParameter<bool>("verbose", false);
  publishFrequency_ = pset.getUntrackedParameter<double>("publishFrequency", publishFrequency_);
  std::string filter = pset.getUntrackedParameter<std::string>("filter", "");

  if (host != "" && port > 0)
  {
    net_ = new DQMBasicNet;
    net_->debug(verbose);
    net_->updateToCollector(host, port);
    net_->start();
  }

  if (! filter.empty())
  {
    try
    {
      filter_ = new lat::Regexp(filter);
      if (! filter_->valid())
	throw cms::Exception("DQMService")
	  << "Invalid 'filter' parameter value '" << filter << "':"
	  << " bad regular expression syntax at character "
	  << filter_->errorOffset() << ": " << filter_->errorMessage();
      filter_->study();
    }
    catch (lat::Error &e)
    {
      throw cms::Exception("DQMService")
	<< "Invalid regular expression 'filter' parameter value '"
	<< filter << "': " << e.explain();
    }
  }
}

DQMService::~DQMService(void)
{
  shutdown();
}

// Flush updates to the network layer at the end of each event.  This
// is the only point at which the main application and the network
// layer interact outside initialisation and exit.
void DQMService::flushStandalone()
{
  // Avoid sending updates excessively often.
  uint64_t version = lat::Time::current().ns();
  double vtime = version * 1e-9;
  if (vtime - lastFlush_ < publishFrequency_)
    return;

  // OK, send an update.
  if (net_)
  {
    DQMNet::Object o;
    std::set<std::string> seen;
    std::string fullpath;

    // Lock the network layer so we can modify the data.
    net_->lock();
    bool updated = false;

    // Find updated contents and update the network cache.
    DQMStore::MEMap::iterator i, e;
    net_->reserveLocalSpace(store_->data_.size());
    for (i = store_->data_.begin(), e = store_->data_.end(); i != e; ++i)
    {
      const MonitorElement &me = *i;
      fullpath.clear();
      fullpath += *me.data_.dirname;
      if (! me.data_.dirname->empty())
        fullpath += '/';
      fullpath += me.data_.objname;

      if (filter_ && filter_->search(fullpath) < 0)
	continue;

      seen.insert(fullpath);
      if (! me.wasUpdated())
	continue;

      o.lastreq = 0;
      o.hash = DQMNet::dqmhash(fullpath.c_str(), fullpath.size());
      o.flags = me.data_.flags;
      o.tag = me.data_.tag;
      o.version = version;
      o.dirname = me.data_.dirname;
      o.objname = me.data_.objname;
      assert(o.rawdata.empty());
      assert(o.scalar.empty());
      assert(o.qdata.empty());

      // Pack object and reference, scalar and quality data.
      switch (me.kind())
      {
      case MonitorElement::DQM_KIND_INT:
      case MonitorElement::DQM_KIND_REAL:
      case MonitorElement::DQM_KIND_STRING:
	me.packScalarData(o.scalar, "");
	break;

      default:
	{
          TBufferFile buffer(TBufferFile::kWrite);
          buffer.WriteObject(me.object_);
          if (me.reference_)
	    buffer.WriteObject(me.reference_);
          else
	    buffer.WriteObjectAny(0, 0);
          o.rawdata.resize(buffer.Length());
          memcpy(&o.rawdata[0], buffer.Buffer(), buffer.Length());
          DQMNet::packQualityData(o.qdata, me.data_.qreports);
	  break;
	}
      }

      // Update.
      net_->updateLocalObject(o);
      DQMNet::DataBlob().swap(o.rawdata);
      std::string().swap(o.scalar);
      std::string().swap(o.qdata);
      updated = true;
    }

    // Find removed contents and clear the network cache.
    if (net_->removeLocalExcept(seen))
      updated = true;

    // Unlock the network layer.
    net_->unlock();

    // Tell network to flush if we updated something.
    if (updated)
      net_->sendLocalChanges();
  }

  store_->reset();
  lastFlush_ = lat::Time::current().ns() * 1e-9;


}
void
DQMService::flush(const edm::Event &, const edm::EventSetup &)
{
  // Call a function independent to the framework
  flushStandalone();
}

// Disengage the network service.
void
DQMService::shutdown(void)
{
  // If we have a network, let it go.
  if (net_)
    net_->shutdown();
}

#endif // !WITHOUT_CMS_FRAMEWORK
