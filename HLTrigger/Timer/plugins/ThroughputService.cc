// C++ headers
#include <algorithm>
#include <chrono>

// boost headers
#include <boost/format.hpp>

// CMSSW headers
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "ThroughputService.h"

// describe the module's configuration
void ThroughputService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<double>(        "timeRange",       60000.0 );
  desc.addUntracked<double>(        "timeResolution",     10.0 );
  desc.addUntracked<std::string>(   "dqmPath",           "HLT/Throughput" );
  descriptions.add("ThroughputService", desc);
}

ThroughputService::ThroughputService(const edm::ParameterSet & config, edm::ActivityRegistry & registry) :
  m_stream_histograms(),
  // configuration
  m_time_range(      config.getUntrackedParameter<double>("timeRange") ),
  m_time_resolution( config.getUntrackedParameter<double>("timeResolution") ),
  m_dqm_path(        config.getUntrackedParameter<std::string>("dqmPath" ) )
{
  registry.watchPreallocate(        this, & ThroughputService::preallocate );
  registry.watchPreStreamBeginRun(  this, & ThroughputService::preStreamBeginRun );
  registry.watchPostStreamEndLumi(  this, & ThroughputService::postStreamEndLumi );
  registry.watchPostStreamEndRun(   this, & ThroughputService::postStreamEndRun );
  registry.watchPreSourceEvent(     this, & ThroughputService::preSourceEvent );
  registry.watchPostEvent(          this, & ThroughputService::postEvent );
}

ThroughputService::~ThroughputService()
{
}

void
ThroughputService::preallocate(edm::service::SystemBounds const & bounds)
{
  m_startup = std::chrono::steady_clock::now();

  m_stream_histograms.resize( bounds.maxNumberOfStreams() );

  // assign a pseudo module id to the FastTimerService
  m_module_id = edm::ModuleDescription::getUniqueID();
}

void
ThroughputService::preStreamBeginRun(edm::StreamContext const & sc)
{
  // if the DQMStore is available, book the DQM histograms
  if (edm::Service<DQMStore>().isAvailable()) {
    unsigned int sid = sc.streamID().value();
    auto & stream = m_stream_histograms[sid];

    std::string   y_axis_title = (boost::format("events / %g s") % m_time_resolution).str();
    unsigned int  bins         = std::round( m_time_range / m_time_resolution );
    double        range        = bins * m_time_resolution; 

    // define a callback that can book the histograms
    auto bookTransactionCallback = [&, this] (DQMStore::IBooker & booker) {
      booker.setCurrentFolder(m_dqm_path);
      stream.sourced_events = booker.book1D("throughput_sourced",  "Throughput (sourced events)",   bins, 0., range)->getTH1F();
      stream.sourced_events ->SetXTitle("time [s]");
      stream.sourced_events ->SetYTitle(y_axis_title.c_str());
      stream.retired_events = booker.book1D("throughput_retired",  "Throughput (retired events)",   bins, 0., range)->getTH1F();
      stream.retired_events ->SetXTitle("time [s]");
      stream.retired_events ->SetYTitle(y_axis_title.c_str());
    };

    // book MonitorElement's for this stream
    edm::Service<DQMStore>()->bookTransaction(bookTransactionCallback, sc.eventID().run(), sid, m_module_id);
  }
}

void
ThroughputService::postStreamEndLumi(edm::StreamContext const& sc)
{
  if (edm::Service<DQMStore>().isAvailable())
    edm::Service<DQMStore>()->mergeAndResetMEsLuminositySummaryCache(sc.eventID().run(), sc.eventID().luminosityBlock(), sc.streamID().value(), m_module_id);
}

void
ThroughputService::postStreamEndRun(edm::StreamContext const & sc)
{
  if (edm::Service<DQMStore>().isAvailable())
    edm::Service<DQMStore>()->mergeAndResetMEsRunSummaryCache(sc.eventID().run(), sc.streamID().value(), m_module_id);
}

void
ThroughputService::preSourceEvent(edm::StreamID sid)
{
  auto timestamp = std::chrono::steady_clock::now();
  m_stream_histograms[sid].sourced_events->Fill( std::chrono::duration_cast<std::chrono::duration<double>>(timestamp - m_startup).count() );
}

void
ThroughputService::postEvent(edm::StreamContext const & sc)
{
  unsigned int sid = sc.streamID().value();
  auto timestamp = std::chrono::steady_clock::now();
  m_stream_histograms[sid].retired_events->Fill( std::chrono::duration_cast<std::chrono::duration<double>>(timestamp - m_startup).count() );
}


// declare ThroughputService as a framework Service
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(ThroughputService);
