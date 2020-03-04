// C++ headers
#include <algorithm>
#include <chrono>

// boost headers
#include <boost/format.hpp>

// CMSSW headers
#include "DQMServices/Core/interface/DQMStore.h"
#include "ThroughputService.h"

// local headers
#include "processor_model.h"

// describe the module's configuration
void ThroughputService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<double>("timeRange", 60000.0);
  desc.addUntracked<double>("timeResolution", 10.0);
  desc.addUntracked<std::string>("dqmPath", "HLT/Throughput");
  desc.addUntracked<bool>("dqmPathByProcesses", false);
  descriptions.add("ThroughputService", desc);
}

ThroughputService::ThroughputService(const edm::ParameterSet& config, edm::ActivityRegistry& registry)
    :  // startup time
      m_startup(std::chrono::steady_clock::now()),
      // configuration
      m_time_range(config.getUntrackedParameter<double>("timeRange")),
      m_time_resolution(config.getUntrackedParameter<double>("timeResolution")),
      m_dqm_path(config.getUntrackedParameter<std::string>("dqmPath")),
      m_dqm_bynproc(config.getUntrackedParameter<bool>("dqmPathByProcesses")) {
  registry.watchPreGlobalBeginRun(this, &ThroughputService::preGlobalBeginRun);
  registry.watchPreSourceEvent(this, &ThroughputService::preSourceEvent);
  registry.watchPostEvent(this, &ThroughputService::postEvent);
}

ThroughputService::~ThroughputService() = default;

void ThroughputService::preallocate(edm::service::SystemBounds const& bounds) {
  auto concurrent_streams = bounds.maxNumberOfStreams();
  auto concurrent_threads = bounds.maxNumberOfThreads();

  if (m_dqm_bynproc)
    m_dqm_path += (boost::format("/Running on %s with %d streams on %d threads") % processor_model %
                   concurrent_streams % concurrent_threads)
                      .str();
}

void ThroughputService::preGlobalBeginRun(edm::GlobalContext const& gc) {
  // if the DQMStore is available, book the DQM histograms
  if (edm::Service<DQMStore>().isAvailable()) {
    std::string y_axis_title = (boost::format("events / %g s") % m_time_resolution).str();
    unsigned int bins = std::round(m_time_range / m_time_resolution);
    double range = bins * m_time_resolution;

    // define a callback that can book the histograms
    auto bookTransactionCallback = [&, this](DQMStore::IBooker& booker, DQMStore::IGetter&) {
      booker.setCurrentFolder(m_dqm_path);
      m_sourced_events = booker.book1D("throughput_sourced", "Throughput (sourced events)", bins, 0., range);
      m_sourced_events->setXTitle("time [s]");
      m_sourced_events->setYTitle(y_axis_title);
      m_retired_events = booker.book1D("throughput_retired", "Throughput (retired events)", bins, 0., range);
      m_retired_events->setXTitle("time [s]");
      m_retired_events->setYTitle(y_axis_title);
    };

    // book MonitorElement's for this run
    edm::Service<DQMStore>()->meBookerGetter(bookTransactionCallback);
  } else {
    std::cerr << "No DQMStore service, aborting." << std::endl;
    abort();
  }
}

void ThroughputService::preSourceEvent(edm::StreamID sid) {
  auto timestamp = std::chrono::steady_clock::now();
  m_sourced_events->Fill(std::chrono::duration_cast<std::chrono::duration<double>>(timestamp - m_startup).count());
}

void ThroughputService::postEvent(edm::StreamContext const& sc) {
  auto timestamp = std::chrono::steady_clock::now();
  m_retired_events->Fill(std::chrono::duration_cast<std::chrono::duration<double>>(timestamp - m_startup).count());
}

// declare ThroughputService as a framework Service
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(ThroughputService);
