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
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// helper functions
template <typename T>
static
const T * get(const edm::Event & event, const edm::EDGetTokenT<T> & token) {
  edm::Handle<T> handle;
  event.getByToken(token, handle);
  if (not handle.isValid())
    throw * handle.whyFailed();
  return handle.product();
}

template <typename R, typename T>
static
const T * get(const edm::EventSetup & setup) {
  edm::ESHandle<T> handle;
  setup.get<R>().get(handle);
  return handle.product();
}


class TriggerBxMonitor : public DQMEDAnalyzer {
public:
  explicit TriggerBxMonitor(edm::ParameterSet const &);
  ~TriggerBxMonitor();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  virtual void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(edm::Event const &, edm::EventSetup const &) override;

  // number of bunch crossings
  static const unsigned int s_bx_range = 4000;

  // module configuration
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord>    m_l1t_results;
  edm::EDGetTokenT<edm::TriggerResults>             m_hlt_results;
  std::string                                       m_dqm_path;

  L1GtTriggerMenu const * m_l1tMenu;
  L1GtTriggerMask const * m_l1tAlgoMask;
  L1GtTriggerMask const * m_l1tTechMask;
  HLTConfigProvider       m_hltConfig;

  // L1T triggers
  std::vector<TH1F *>     m_l1t_algo_bx;
  std::vector<TH1F *>     m_l1t_tech_bx;

  // HLT triggers
  std::vector<TH1F *>     m_hlt_bx;
};



void TriggerBxMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>( "l1tResults",       edm::InputTag("gtDigis"));
  desc.addUntracked<edm::InputTag>( "hltResults",       edm::InputTag("TriggerResults"));
  desc.addUntracked<std::string>(   "dqmPath",          "HLT/TriggerBx" );
  descriptions.add("triggerBxMonitor", desc);
}


TriggerBxMonitor::TriggerBxMonitor(edm::ParameterSet const & config) :
  // module configuration
  m_l1t_results( consumes<L1GlobalTriggerReadoutRecord>( config.getUntrackedParameter<edm::InputTag>( "l1tResults" ) ) ),
  m_hlt_results( consumes<edm::TriggerResults>(          config.getUntrackedParameter<edm::InputTag>( "hltResults" ) ) ),
  m_dqm_path(                                            config.getUntrackedParameter<std::string>(   "dqmPath" ) ),
  // L1T and HLT configuration
  m_l1tMenu( nullptr ),
  m_l1tAlgoMask( nullptr ),
  m_l1tTechMask( nullptr),
  m_hltConfig(),
  // L1T triggers
  m_l1t_algo_bx(),
  m_l1t_tech_bx(),
  // HLT triggers
  m_hlt_bx()
{
}

TriggerBxMonitor::~TriggerBxMonitor()
{
}

void TriggerBxMonitor::dqmBeginRun(edm::Run const & run, edm::EventSetup const & setup)
{
  // cache the L1 trigger menu
  m_l1tMenu     = get<L1GtTriggerMenuRcd, L1GtTriggerMenu>(setup);
  m_l1tAlgoMask = get<L1GtTriggerMaskAlgoTrigRcd, L1GtTriggerMask>(setup);
  m_l1tTechMask = get<L1GtTriggerMaskTechTrigRcd, L1GtTriggerMask>(setup);
  if (m_l1tMenu and m_l1tAlgoMask and m_l1tTechMask) {
    m_l1t_algo_bx.clear();
    m_l1t_algo_bx.resize( m_l1tAlgoMask->gtTriggerMask().size(), nullptr );
    m_l1t_tech_bx.clear();
    m_l1t_tech_bx.resize( m_l1tTechMask->gtTriggerMask().size(), nullptr );
  } else {
    // L1GtUtils not initialised, skip the the L1T monitoring
    edm::LogError("TriggerBxMonitor") << "failed to read the L1 menu or masks from the EventSetup, the L1 trigger bx distribution  will not be monitored";
  }

  // initialise the HLTConfigProvider
  bool changed = true;
  edm::EDConsumerBase::Labels labels;
  labelsForToken(m_hlt_results, labels);
  if (m_hltConfig.init(run, setup, labels.process, changed)) {
    m_hlt_bx.clear();
    m_hlt_bx.resize( m_hltConfig.size(), nullptr );
  } else {
    // HLTConfigProvider not initialised, skip the the HLT monitoring
    edm::LogError("TriggerBxMonitor") << "failed to initialise HLTConfigProvider, the HLT bx distribution will not be monitored";
  }
}

void TriggerBxMonitor::bookHistograms(DQMStore::IBooker & booker, edm::Run const & run, edm::EventSetup const & setup)
{
  // book the overall event count and event types histograms
  booker.setCurrentFolder( m_dqm_path );

  if (m_l1tMenu and m_l1tAlgoMask) {
    // book the rate histograms for the L1 Algorithm triggers
    booker.setCurrentFolder( m_dqm_path + "/L1 Algo" );

    // book the histograms for L1 algo triggers that are included in the L1 menu
    for (auto const & keyval: m_l1tMenu->gtAlgorithmAliasMap()) {
      int bit = keyval.second.algoBitNumber();
      std::string const & name  = (boost::format("%s (bit %d)") % keyval.first.substr(0, keyval.first.find_first_of(".")) % bit).str();
      m_l1t_algo_bx.at(bit) = booker.book1D(name, name, s_bx_range + 1, -0.5, s_bx_range + 0.5)->getTH1F();
    }
    // book the histograms for L1 algo triggers that are not included in the L1 menu
    for (unsigned int bit = 0; bit < m_l1tAlgoMask->gtTriggerMask().size(); ++bit) if (not m_l1t_algo_bx.at(bit)) {
      std::string const & name  = (boost::format("L1 Algo (bit %d)") % bit).str();
      m_l1t_algo_bx.at(bit) = booker.book1D(name, name, s_bx_range + 1, -0.5, s_bx_range + 0.5)->getTH1F();
    }
  }

  if (m_l1tMenu and m_l1tTechMask) {
    // book the rate histograms for the L1 Technical triggers
    booker.setCurrentFolder( m_dqm_path + "/L1 Tech" );

    // book the histograms for L1 tech triggers that are included in the L1 menu
    for (auto const & keyval: m_l1tMenu->gtTechnicalTriggerMap()) {
      int bit = keyval.second.algoBitNumber();
      std::string const & name  = (boost::format("%s (bit %d)") % keyval.first.substr(0, keyval.first.find_first_of(".")) % bit).str();
      m_l1t_tech_bx.at(bit) = booker.book1D(name, name, s_bx_range + 1, -0.5, s_bx_range + 0.5)->getTH1F();
    }
    // book the histograms for L1 tech triggers that are not included in the L1 menu
    for (unsigned int bit = 0; bit < m_l1tTechMask->gtTriggerMask().size(); ++bit) if (not m_l1t_tech_bx.at(bit)) {
      std::string const & name  = (boost::format("L1 Tech (bit %d)") % bit).str();
      m_l1t_tech_bx.at(bit) = booker.book1D(name, name, s_bx_range + 1, -0.5, s_bx_range + 0.5)->getTH1F();
    }

  }

  if (m_hltConfig.inited()) {
    // book the HLT triggers rate histograms
    booker.setCurrentFolder( m_dqm_path + "/HLT" );
    for (unsigned int i = 0; i < m_hltConfig.size(); ++i) {
      std::string const & name = m_hltConfig.triggerName(i);
      m_hlt_bx[i] = booker.book1D(name, name, s_bx_range + 1, -0.5, s_bx_range + 0.5)->getTH1F();
    }
  }
}


void TriggerBxMonitor::analyze(edm::Event const & event, edm::EventSetup const & setup)
{
  L1GlobalTriggerReadoutRecord const & l1tResults = * get<L1GlobalTriggerReadoutRecord>(event, m_l1t_results);
  unsigned int bx = l1tResults.gtfeWord().bxNr();

  // monitor the bx distribution for the L1 triggers
  if (m_l1tMenu) {
    const std::vector<bool> & algoword = l1tResults.decisionWord();
    if (algoword.size() == m_l1t_algo_bx.size()) {
      for (unsigned int i = 0; i < m_l1t_algo_bx.size(); ++i)
        if (algoword[i])
          m_l1t_algo_bx[i]->Fill(bx);
    } else {
      edm::LogWarning("TriggerBxMonitor") << "This should never happen: the size of the L1 Algo Trigger mask does not match the number of L1 Algo Triggers";
    }

    const std::vector<bool> & techword = l1tResults.technicalTriggerWord();
    if (techword.size() == m_l1t_tech_bx.size()) {
      for (unsigned int i = 0; i < m_l1t_tech_bx.size(); ++i)
        if (techword[i])
          m_l1t_tech_bx[i]->Fill(bx);
    } else {
      edm::LogWarning("TriggerBxMonitor") << "This should never happen: the size of the L1 Tech Trigger mask does not match the number of L1 Tech Triggers";
    }
  }

  // monitor the bx distribution for the HLT triggers
  if (m_hltConfig.inited()) {
    edm::TriggerResults const & hltResults = * get<edm::TriggerResults>(event, m_hlt_results);
    if (hltResults.size() == m_hlt_bx.size()) {
      for (unsigned int i = 0; i < m_hlt_bx.size(); ++i) {
        if (hltResults.at(i).accept())
          m_hlt_bx[i]->Fill(bx);
      }
    } else {
      edm::LogWarning("TriggerBxMonitor") << "This should never happen: the number of HLT paths has changed since the beginning of the run";
    }
  }
}


//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerBxMonitor);
