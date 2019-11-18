#if !WITHOUT_CMS_FRAMEWORK
#include "DQMServices/Core/src/DQMService.h"
#include "DQMServices/Core/interface/DQMNet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMScope.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "classlib/utils/Regexp.h"
#include "classlib/utils/Error.h"
#include <mutex>
#include <iostream>
#include <string>
#include <memory>
#include "TBufferFile.h"

// -------------------------------------------------------------------
static std::recursive_mutex s_mutex;

/// Acquire lock and access to the DQM core from a thread other than
/// the "main" CMSSW processing thread, such as in extra XDAQ threads.
DQMScope::DQMScope() { s_mutex.lock(); }

/// Release access lock to the DQM core.
DQMScope::~DQMScope() { s_mutex.unlock(); }

// -------------------------------------------------------------------
DQMService::DQMService(const edm::ParameterSet &pset, edm::ActivityRegistry &ar)
    : store_(&*edm::Service<DQMStore>()), net_(nullptr), filter_(nullptr), lastFlush_(0), publishFrequency_(5.0) {
  ar.watchPostEvent(this, &DQMService::flush);
  ar.watchPostStreamEndLumi(this, &DQMService::flush);

  std::string host = pset.getUntrackedParameter<std::string>("collectorHost", "");
  int port = pset.getUntrackedParameter<int>("collectorPort", 9090);
  bool verbose = pset.getUntrackedParameter<bool>("verbose", false);
  publishFrequency_ = pset.getUntrackedParameter<double>("publishFrequency", publishFrequency_);
  std::string filter = pset.getUntrackedParameter<std::string>("filter", "");

  if (!host.empty() && port > 0) {
    net_ = new DQMBasicNet;
    net_->debug(verbose);
    net_->updateToCollector(host, port);
    net_->start();
  }

  if (!filter.empty()) {
    try {
      filter_ = new lat::Regexp(filter);
      if (!filter_->valid())
        throw cms::Exception("DQMService") << "Invalid 'filter' parameter value '" << filter << "':"
                                           << " bad regular expression syntax at character " << filter_->errorOffset()
                                           << ": " << filter_->errorMessage();
      filter_->study();
    } catch (lat::Error &e) {
      throw cms::Exception("DQMService") << "Invalid regular expression 'filter' parameter value '" << filter
                                         << "': " << e.explain();
    }
  }
}

DQMService::~DQMService() { shutdown(); }

// Flush updates to the network layer at the end of each event.  This
// is the only point at which the main application and the network
// layer interact outside initialisation and exit.
void DQMService::flushStandalone() {
  // Avoid sending updates excessively often.
  uint64_t version = lat::Time::current().ns();
  double vtime = version * 1e-9;
  if (vtime - lastFlush_ < publishFrequency_)
    return;

  // OK, send an update.
  if (net_) {
    DQMNet::Object o;
    std::set<std::string> seen;
    std::string fullpath;

    // Lock the network layer so we can modify the data.
    net_->lock();
    bool updated = false;

    // TODO: re-implement sending MEs.

    // Find removed contents and clear the network cache.
    if (net_->removeLocalExcept(seen))
      updated = true;

    // Unlock the network layer.
    net_->unlock();

    // Tell network to flush if we updated something.
    if (updated)
      net_->sendLocalChanges();
  }

  lastFlush_ = lat::Time::current().ns() * 1e-9;
}
void DQMService::flush(edm::StreamContext const &sc) {
  // Call a function independent to the framework
  flushStandalone();
}

// Disengage the network service.
void DQMService::shutdown() {
  // If we have a network, let it go.
  if (net_)
    net_->shutdown();
}

#endif  // !WITHOUT_CMS_FRAMEWORK
