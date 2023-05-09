#ifndef ThroughputService_h
#define ThroughputService_h

// C++ headers
#include <atomic>
#include <chrono>
#include <functional>
#include <string>

// TBB headers
#include <oneapi/tbb/concurrent_vector.h>

// ROOT headers
#include <TH1F.h>

// CMSSW headers
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"

class ThroughputService {
public:
  typedef dqm::reco::DQMStore DQMStore;

  ThroughputService(const edm::ParameterSet&, edm::ActivityRegistry&);
  ~ThroughputService() = default;

private:
  void preallocate(edm::service::SystemBounds const& bounds);
  void preGlobalBeginRun(edm::GlobalContext const& gc);
  void preSourceEvent(edm::StreamID sid);
  void postEvent(edm::StreamContext const& sc);
  void postEndJob();

public:
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  dqm::reco::MonitorElement* m_sourced_events = nullptr;
  dqm::reco::MonitorElement* m_retired_events = nullptr;

  std::chrono::system_clock::time_point m_startup;

  // event time buffer
  const uint32_t m_resolution;
  std::atomic<uint32_t> m_counter;
  tbb::concurrent_vector<std::chrono::system_clock::time_point> m_events;
  bool m_print_event_summary;

  // DQM related data members
  bool m_enable_dqm;
  const bool m_dqm_bynproc;
  std::string m_dqm_path;
  const double m_time_range;
  const double m_time_resolution;
};

#endif  // ! ThroughputService_h
