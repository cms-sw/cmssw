// Note to self: the implementation uses TH1F's to store the L1 and HLT rates.
// Assuming a maximum rate of 100 kHz times a period of 23.31 s, one needs to store counts up to ~2.3e6.
// A "float" has 24 bits of precision, so it can store up to 2**24 ~ 16.7e6 without loss of precision.


// C++ headers
#include <string>
#include <cstring>

// boost headers
#include <boost/regex.hpp>
#include <boost/format.hpp>

// Root headers
#include <TH1F.h>

// CMSSW headers
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// helper functions
template <typename T>
static
const T & get(const edm::Event & event, const edm::EDGetTokenT<T> & token) {
  edm::Handle<T> handle;
  event.getByToken(token, handle);
  if (not handle.isValid())
    throw * handle.whyFailed();
  return * handle.product();
}

template <typename R, typename T>
static
const T & get(const edm::EventSetup & setup) {
  edm::ESHandle<T> handle;
  setup.get<R>().get(handle);
  return * handle.product();
}


class TriggerRatesMonitor : public DQMEDAnalyzer {
public:
  explicit TriggerRatesMonitor(edm::ParameterSet const &);
  ~TriggerRatesMonitor();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  virtual void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(edm::Event const &, edm::EventSetup const &) override;

  // TCDS trigger types
  // see https://twiki.cern.ch/twiki/bin/viewauth/CMS/TcdsEventRecord
  static constexpr const char * const s_tcds_trigger_types[] = {
    "Empty",           //  0 - No trigger
    "Physics",         //  1 - GT trigger
    "Calibration",     //  2 - Sequence trigger (calibration)
    "Random",          //  3 - Random trigger
    "Auxiliary",       //  4 - Auxiliary (CPM front panel NIM input) trigger
    nullptr,           //  5 - reserved
    nullptr,           //  6 - reserved
    nullptr,           //  7 - reserved
    "Cyclic",          //  8 - Cyclic trigger
    "Bunch-pattern",   //  9 - Bunch-pattern trigger
    "Software",        // 10 - Software trigger
    "TTS",             // 11 - TTS-sourced trigger
    nullptr,           // 12 - reserved
    nullptr,           // 13 - reserved
    nullptr,           // 14 - reserved
    nullptr            // 15 - reserved
  };

  // module configuration
  const edm::EDGetTokenT<GlobalAlgBlkBxCollection>  m_l1t_results;
  const edm::EDGetTokenT<edm::TriggerResults>       m_hlt_results;
  std::string                                       m_dqm_path;
  uint32_t                                          m_lumisections_range;

  // L1T and HLT configuration

  L1TUtmTriggerMenu const * m_l1tMenu;

  struct HLTIndices {
    unsigned int index_l1_seed;
    unsigned int index_prescale;

    HLTIndices() :
      index_l1_seed(  (unsigned int) -1),
      index_prescale( (unsigned int) -1)
    { }
  };

  HLTConfigProvider             m_hltConfig;
  std::vector<HLTIndices>       m_hltIndices;

  std::vector<std::vector<unsigned int>> m_datasets;
  std::vector<std::vector<unsigned int>> m_streams;

  // L1T and HLT rate plots

  struct HLTRatesPlots {
    TH1F * pass_l1_seed;
    TH1F * pass_prescale;
    TH1F * accept;
    TH1F * reject;
    TH1F * error;
  };

  // overall event count and event types
  TH1F *                        m_events_processed;
  std::vector<TH1F *>           m_tcds_counts;

  // L1T triggers
  std::vector<TH1F *>           m_l1t_counts;

  // HLT triggers
  std::vector<HLTRatesPlots>    m_hlt_counts;

  // datasets
  std::vector<TH1F *>           m_dataset_counts;

  // streams
  std::vector<TH1F *>           m_stream_counts;

};

// definition
constexpr const char * const TriggerRatesMonitor::s_tcds_trigger_types[];


void TriggerRatesMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>( "l1tResults",       edm::InputTag("gtStage2Digis"));
  desc.addUntracked<edm::InputTag>( "hltResults",       edm::InputTag("TriggerResults"));
  desc.addUntracked<std::string>(   "dqmPath",          "HLT/TriggerRates" );
  desc.addUntracked<uint32_t>(      "lumisectionRange", 2500 );             // ~16 hours
  descriptions.add("triggerRatesMonitor", desc);
}


TriggerRatesMonitor::TriggerRatesMonitor(edm::ParameterSet const & config) :
  // module configuration
  m_l1t_results(consumes<GlobalAlgBlkBxCollection>( config.getUntrackedParameter<edm::InputTag>( "l1tResults" ) )),
  m_hlt_results(consumes<edm::TriggerResults>(      config.getUntrackedParameter<edm::InputTag>( "hltResults" ) )),
  m_dqm_path(                                       config.getUntrackedParameter<std::string>(   "dqmPath" ) ),
  m_lumisections_range(                             config.getUntrackedParameter<uint32_t>(      "lumisectionRange" ) ),
  // L1T and HLT configuration
  m_l1tMenu(nullptr),
  m_hltConfig(),
  m_hltIndices(),
  m_datasets(),
  m_streams(),
  // overall event count and event types
  m_events_processed(nullptr),
  m_tcds_counts(),
  // L1T triggers
  m_l1t_counts(),
  // HLT triggers
  m_hlt_counts(),
  // datasets
  m_dataset_counts(),
  // streams
  m_stream_counts()
{
}

TriggerRatesMonitor::~TriggerRatesMonitor()
{
}

void TriggerRatesMonitor::dqmBeginRun(edm::Run const & run, edm::EventSetup const & setup)
{
  m_events_processed = nullptr;
  m_tcds_counts.clear();
  m_tcds_counts.resize(sizeof(s_tcds_trigger_types)/sizeof(const char *), nullptr);

  // cache the L1 trigger menu
  m_l1tMenu = & get<L1TUtmTriggerMenuRcd, L1TUtmTriggerMenu>(setup);
  if (m_l1tMenu) {
    m_l1t_counts.clear();
    m_l1t_counts.resize(GlobalAlgBlk::maxPhysicsTriggers, nullptr);
  } else {
    edm::LogError("TriggerRatesMonitor") << "failed to read the L1 menu from the EventSetup, the L1 trigger rates will not be monitored";
  }

  // initialise the HLTConfigProvider
  bool changed = true;
  edm::EDConsumerBase::Labels labels;
  labelsForToken(m_hlt_results, labels);
  if (m_hltConfig.init(run, setup, labels.process, changed)) {
    m_hlt_counts.clear();
    m_hlt_counts.resize( m_hltConfig.size(), HLTRatesPlots() );
    m_hltIndices.resize( m_hltConfig.size(), HLTIndices() );

    unsigned int datasets = m_hltConfig.datasetNames().size();
    m_datasets.clear();
    m_datasets.resize( datasets, {} );
    for (unsigned int i = 0; i < datasets; ++i) {
      auto const & paths = m_hltConfig.datasetContent(i);
      m_datasets[i].reserve(paths.size());
      for (auto const & path: paths)
        m_datasets[i].push_back(m_hltConfig.triggerIndex(path));
    }
    m_dataset_counts.clear();
    m_dataset_counts.resize( datasets, nullptr );

    unsigned int streams = m_hltConfig.streamNames().size();
    m_streams.clear();
    m_streams.resize( streams, {} );
    for (unsigned int i = 0; i < streams; ++i) {
      for (auto const & dataset : m_hltConfig.streamContent(i)) {
        for (auto const & path : m_hltConfig.datasetContent(dataset))
          m_streams[i].push_back(m_hltConfig.triggerIndex(path));
      }
      std::sort(m_streams[i].begin(), m_streams[i].end());
      auto unique_end = std::unique(m_streams[i].begin(), m_streams[i].end());
      m_streams[i].resize(unique_end - m_streams[i].begin());
      m_streams[i].shrink_to_fit();
    }
    m_stream_counts.clear();
    m_stream_counts.resize( streams, nullptr );
  } else {
    // HLTConfigProvider not initialised, skip the the HLT monitoring
    edm::LogError("TriggerRatesMonitor") << "failed to initialise HLTConfigProvider, the HLT trigger and datasets rates will not be monitored";
  }
}

void TriggerRatesMonitor::bookHistograms(DQMStore::IBooker & booker, edm::Run const & run, edm::EventSetup const & setup)
{
  // book the overall event count and event types histograms
  booker.setCurrentFolder( m_dqm_path );
  m_events_processed = booker.book1D("events", "Processed events vs. lumisection", m_lumisections_range + 1, -0.5, m_lumisections_range + 0.5)->getTH1F();
  booker.setCurrentFolder( m_dqm_path + "/TCDS" );
  for (unsigned int i = 0; i < sizeof(s_tcds_trigger_types)/sizeof(const char *); ++i)
    if (s_tcds_trigger_types[i]) {
      std::string const & title = (boost::format("%s events vs. lumisection") % s_tcds_trigger_types[i]).str();
      m_tcds_counts[i] = booker.book1D(s_tcds_trigger_types[i], title, m_lumisections_range + 1, -0.5, m_lumisections_range + 0.5)->getTH1F();
    }

  if (m_l1tMenu) {
    // book the rate histograms for the L1 triggers that are included in the L1 menu
    booker.setCurrentFolder( m_dqm_path + "/L1T" );
    for (auto const & keyval: m_l1tMenu->getAlgorithmMap()) {
      unsigned int bit = keyval.second.getIndex();
      bool masked = false;      // FIXME read L1 masks once they will be avaiable in the EventSetup
      std::string const & name  = (boost::format("%s (bit %d)") % keyval.first % bit).str();
      std::string const & title = (boost::format("%s (bit %d)%s vs. lumisection") % keyval.first % bit % (masked ? " (masked)" : "")).str();
      m_l1t_counts.at(bit) = booker.book1D(name, title, m_lumisections_range + 1, -0.5, m_lumisections_range + 0.5)->getTH1F();
    }
  }

  if (m_hltConfig.inited()) {
    // book the rate histograms for the HLT triggers
    booker.setCurrentFolder( m_dqm_path + "/HLT" );
    for (unsigned int i = 0; i < m_hltConfig.size(); ++i) {
      std::string const & name = m_hltConfig.triggerName(i);
      m_hlt_counts[i].pass_l1_seed  = booker.book1D(name + " pass L1 seed",     name + " pass L1 seed, vs. lumisection",     m_lumisections_range + 1,   -0.5,   m_lumisections_range + 0.5)->getTH1F();
      m_hlt_counts[i].pass_prescale = booker.book1D(name + " pass prescaler",   name + " pass prescaler, vs. lumisection",   m_lumisections_range + 1,   -0.5,   m_lumisections_range + 0.5)->getTH1F();
      m_hlt_counts[i].accept        = booker.book1D(name + " accept",           name + " accept, vs. lumisection",           m_lumisections_range + 1,   -0.5,   m_lumisections_range + 0.5)->getTH1F();
      m_hlt_counts[i].reject        = booker.book1D(name + " reject",           name + " reject, vs. lumisection",           m_lumisections_range + 1,   -0.5,   m_lumisections_range + 0.5)->getTH1F();
      m_hlt_counts[i].error         = booker.book1D(name + " error",            name + " error, vs. lumisection",            m_lumisections_range + 1,   -0.5,   m_lumisections_range + 0.5)->getTH1F();
      // look for the index of the (last) L1 seed and prescale module in each path
      m_hltIndices[i].index_l1_seed  = m_hltConfig.size(i);
      m_hltIndices[i].index_prescale = m_hltConfig.size(i);
      for (unsigned int j = 0; j < m_hltConfig.size(i); ++j) {
        std::string const & label = m_hltConfig.moduleLabel(i, j);
        std::string const & type  = m_hltConfig.moduleType(label);
        if (type == "HLTL1TSeed" or type == "HLTLevel1GTSeed" or type == "HLTLevel1Activity" or type == "HLTLevel1Pattern") {
          // there might be more L1 seed filters in sequence
          // keep looking and store the index of the last one
          m_hltIndices[i].index_l1_seed  = j;
        } else if (type == "HLTPrescaler") {
          // there should be only one prescaler in a path, and it should follow all L1 seed filters
          m_hltIndices[i].index_prescale = j;
          break;
        }

      }
    }

    // book the HLT datasets rate histograms
    booker.setCurrentFolder( m_dqm_path + "/Datasets" );
    auto const & datasets = m_hltConfig.datasetNames();
    for (unsigned int i = 0; i < datasets.size(); ++i)
      m_dataset_counts[i] = booker.book1D(datasets[i], datasets[i], m_lumisections_range + 1, -0.5, m_lumisections_range + 0.5)->getTH1F();

    // book the HLT streams rate histograms
    booker.setCurrentFolder( m_dqm_path + "/Streams" );
    auto const & streams = m_hltConfig.streamNames();
    for (unsigned int i = 0; i < streams.size(); ++i)
      m_stream_counts[i]  = booker.book1D(streams[i],  streams[i],  m_lumisections_range + 1, -0.5, m_lumisections_range + 0.5)->getTH1F();
  }
}


void TriggerRatesMonitor::analyze(edm::Event const & event, edm::EventSetup const & setup)
{
  unsigned int lumisection = event.luminosityBlock();

  // book the overall event count and event types rates
  m_events_processed->Fill(lumisection);
  if (m_tcds_counts[event.experimentType()])
    m_tcds_counts[event.experimentType()]->Fill(lumisection);

  // monitor the L1 triggers rates
  if (m_l1tMenu) {
    auto const & results = get<GlobalAlgBlkBxCollection>(event, m_l1t_results).at(0, 0);
    for (unsigned int i = 0; i < GlobalAlgBlk::maxPhysicsTriggers; ++i)
      if (results.getAlgoDecisionFinal(i))
        if (m_l1t_counts[i])
          m_l1t_counts[i]->Fill(lumisection);
  }

  // monitor the HLT triggers and datsets rates
  if (m_hltConfig.inited()) {
    edm::TriggerResults const & hltResults = get<edm::TriggerResults>(event, m_hlt_results);
    if (hltResults.size() == m_hlt_counts.size()) {
      for (unsigned int i = 0; i < m_hlt_counts.size(); ++i) {
        edm::HLTPathStatus const & path = hltResults.at(i);
        if (path.index() > m_hltIndices[i].index_l1_seed)
          m_hlt_counts[i].pass_l1_seed->Fill(lumisection);
        if  (path.index() > m_hltIndices[i].index_prescale)
          m_hlt_counts[i].pass_prescale->Fill(lumisection);
        if (path.accept())
          m_hlt_counts[i].accept->Fill(lumisection);
        else if (path.error())
          m_hlt_counts[i].error ->Fill(lumisection);
        else
          m_hlt_counts[i].reject->Fill(lumisection);
      }
    } else {
      edm::LogWarning("TriggerRatesMonitor") << "This should never happen: the number of HLT paths has changed since the beginning of the run";
    }

    for (unsigned int i = 0; i < m_datasets.size(); ++i)
      for (unsigned int j: m_datasets[i])
        if (hltResults.at(j).accept()) {
          m_dataset_counts[i]->Fill(lumisection);
          // ensure each dataset is incremented only once per event
          break;
        }

    for (unsigned int i = 0; i < m_streams.size(); ++i)
      for (unsigned int j: m_streams[i])
        if (hltResults.at(j).accept()) {
          m_stream_counts[i]->Fill(lumisection);
          // ensure each stream is incremented only once per event
          break;
        }
  }
}


//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerRatesMonitor);
