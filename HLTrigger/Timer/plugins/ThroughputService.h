#ifndef ThroughputService_h
#define ThroughputService_h

// C++ headers
#include <string>
#include <chrono>
#include <functional>

// TBB headers
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_unordered_set.h>

// ROOT headers
#include <TH1F.h>

// CMSSW headers
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
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
  ~ThroughputService();

private:
  void preallocate(edm::service::SystemBounds const& bounds);
  void preGlobalBeginRun(edm::GlobalContext const& gc);
  void preSourceEvent(edm::StreamID sid);
  void postEvent(edm::StreamContext const& sc);

public:
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  dqm::reco::MonitorElement* m_sourced_events;
  dqm::reco::MonitorElement* m_retired_events;

  std::chrono::steady_clock::time_point m_startup;

  // histogram-related data members
  const double m_time_range;
  const double m_time_resolution;

  // DQM service-related data members
  std::string m_dqm_path;
  const bool m_dqm_bynproc;
};

#endif  // ! ThroughputService_h
