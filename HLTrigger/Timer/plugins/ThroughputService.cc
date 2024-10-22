// C++ headers
#include <algorithm>
#include <chrono>
#include <ctime>

// {fmt} headers
#include <fmt/printf.h>

// CMSSW headers
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "HLTrigger/Timer/interface/processor_model.h"
#include "ThroughputService.h"

using namespace std::literals;

// describe the module's configuration
void ThroughputService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<uint32_t>("eventRange", 10000)->setComment("Preallocate a buffer for N events");
  desc.addUntracked<uint32_t>("eventResolution", 1)->setComment("Sample the processing time every N events");
  desc.addUntracked<bool>("printEventSummary", false);
  desc.ifValue(edm::ParameterDescription<bool>("enableDQM", true, false),  // "false" means untracked
               // parameters if "enableDQM" is "true"
               true >> (edm::ParameterDescription<bool>("dqmPathByProcesses", false, false) and
                        edm::ParameterDescription<std::string>("dqmPath", "HLT/Throughput", false) and
                        edm::ParameterDescription<double>("timeRange", 60000.0, false) and
                        edm::ParameterDescription<double>("timeResolution", 10.0, false)) or
                   // parameters if "enableDQM" is "false"
                   false >> edm::EmptyGroupDescription());
  descriptions.add("ThroughputService", desc);
}

ThroughputService::ThroughputService(const edm::ParameterSet& config, edm::ActivityRegistry& registry)
    :  // startup time
      m_startup(std::chrono::system_clock::now()),
      // configuration
      m_resolution(config.getUntrackedParameter<uint32_t>("eventResolution")),
      m_counter(0),
      m_events(config.getUntrackedParameter<uint32_t>("eventRange") / m_resolution),  // allocate initial size
      m_print_event_summary(config.getUntrackedParameter<bool>("printEventSummary")),
      m_enable_dqm(config.getUntrackedParameter<bool>("enableDQM")),
      m_dqm_bynproc(m_enable_dqm ? config.getUntrackedParameter<bool>("dqmPathByProcesses") : false),
      m_dqm_path(m_enable_dqm ? config.getUntrackedParameter<std::string>("dqmPath") : ""),
      m_time_range(m_enable_dqm ? config.getUntrackedParameter<double>("timeRange") : 0.),
      m_time_resolution(m_enable_dqm ? config.getUntrackedParameter<double>("timeResolution") : 0.) {
  m_events.clear();  // erases all elements, but does not free internal arrays
  registry.watchPreallocate(this, &ThroughputService::preallocate);
  registry.watchPreGlobalBeginRun(this, &ThroughputService::preGlobalBeginRun);
  registry.watchPreSourceEvent(this, &ThroughputService::preSourceEvent);
  registry.watchPostEvent(this, &ThroughputService::postEvent);
  registry.watchPostEndJob(this, &ThroughputService::postEndJob);
}

void ThroughputService::preallocate(edm::service::SystemBounds const& bounds) {
  auto concurrent_streams = bounds.maxNumberOfStreams();
  auto concurrent_threads = bounds.maxNumberOfThreads();

  if (m_enable_dqm and m_dqm_bynproc)
    m_dqm_path += fmt::sprintf(
        "/Running on %s with %d streams on %d threads", processor_model, concurrent_streams, concurrent_threads);
}

void ThroughputService::preGlobalBeginRun(edm::GlobalContext const& gc) {
  // if the DQMStore is available, book the DQM histograms
  // check that the DQMStore service is available
  if (m_enable_dqm and not edm::Service<DQMStore>().isAvailable()) {
    // the DQMStore is not available, disable all DQM plots
    m_enable_dqm = false;
    edm::LogWarning("ThroughputService") << "The DQMStore is not avalable, the DQM plots will not be generated";
  }

  if (m_enable_dqm) {
    std::string y_axis_title = fmt::sprintf("events / %g s", m_time_resolution);
    unsigned int bins = std::round(m_time_range / m_time_resolution);
    double range = bins * m_time_resolution;

    // clean characters that are deemed unsafe for DQM
    // see the definition of `s_safe` in DQMServices/Core/src/DQMStore.cc
    auto safe_for_dqm = "/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+=_()# "s;
    for (auto& c : m_dqm_path)
      if (safe_for_dqm.find(c) == std::string::npos)
        c = '_';

    // define a callback that can book the histograms
    auto bookTransactionCallback = [&, this](DQMStore::IBooker& booker, DQMStore::IGetter&) {
      auto scope = dqm::reco::DQMStore::IBooker::UseRunScope(booker);
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
    m_sourced_events = nullptr;
    m_retired_events = nullptr;
  }
}

void ThroughputService::preSourceEvent(edm::StreamID sid) {
  auto timestamp = std::chrono::system_clock::now();
  auto interval = std::chrono::duration_cast<std::chrono::duration<double>>(timestamp - m_startup).count();
  if (m_enable_dqm) {
    m_sourced_events->Fill(interval);
  }
}

void ThroughputService::postEvent(edm::StreamContext const& sc) {
  auto timestamp = std::chrono::system_clock::now();
  auto interval = std::chrono::duration_cast<std::chrono::duration<double>>(timestamp - m_startup).count();
  if (m_enable_dqm) {
    m_retired_events->Fill(interval);
  }
  ++m_counter;
  if (m_counter % m_resolution == 0) {
    m_events.push_back(timestamp);
  }
}

void ThroughputService::postEndJob() {
  if (m_counter < 2 * m_resolution) {
    // not enough mesurements to estimate the throughput
    edm::LogWarning("ThroughputService") << "Not enough events to measure the throughput with a resolution of "
                                         << m_resolution << " events";
    return;
  }

  edm::LogInfo info("ThroughputService");

  if (m_print_event_summary) {
    for (uint32_t i = 0; i < m_events.size(); ++i) {
      info << std::setw(8) << (i + 1) * m_resolution << ", " << std::setprecision(6) << edm::TimeOfDay(m_events[i])
           << "\n";
    }
    info << '\n';
  }

  // measure the time to process each block of m_resolution events
  uint32_t blocks = m_counter / m_resolution - 1;
  std::vector<double> delta(blocks);
  for (uint32_t i = 0; i < blocks; ++i) {
    delta[i] = std::chrono::duration_cast<std::chrono::duration<double>>(m_events[i + 1] - m_events[i]).count();
  }
  // measure the average and standard deviation of the time to process m_resolution
  double time_avg = TMath::Mean(delta.begin(), delta.begin() + blocks);
  double time_dev = TMath::StdDev(delta.begin(), delta.begin() + blocks);
  // compute the throughput and its standard deviation across the job
  double throughput_avg = double(m_resolution) / time_avg;
  double throughput_dev = double(m_resolution) * time_dev / time_avg / time_avg;

  info << "Average throughput: " << throughput_avg << " ± " << throughput_dev << " ev/s";
}

// declare ThroughputService as a framework Service
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(ThroughputService);
