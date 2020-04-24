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
#include "FWCore/Framework/interface/EventSetup.h"
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


class TriggerBxMonitor : public DQMEDAnalyzer {
public:
  explicit TriggerBxMonitor(edm::ParameterSet const &);
  ~TriggerBxMonitor() override;

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &, edm::EventSetup const &) override;

  // number of bunch crossings
  static const unsigned int s_bx_range = 3564;

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
  const std::string                                 m_dqm_path;
  const bool                                        m_make_1d_plots;
  const bool                                        m_make_2d_plots;
  const uint32_t                                    m_ls_range;

  // L1T and HLT configuration
  L1TUtmTriggerMenu const * m_l1tMenu;
  HLTConfigProvider         m_hltConfig;

  // L1T and HLT results
  TH2F *                    m_tcds_bx_all;
  TH2F *                    m_l1t_bx_all;
  TH2F *                    m_hlt_bx_all;
  std::vector<TH1F *>       m_tcds_bx;
  std::vector<TH1F *>       m_l1t_bx;
  std::vector<TH1F *>       m_hlt_bx;
  std::vector<TH2F *>       m_tcds_bx_2d;
  std::vector<TH2F *>       m_l1t_bx_2d;
  std::vector<TH2F *>       m_hlt_bx_2d;
};

// definition
constexpr const char * const TriggerBxMonitor::s_tcds_trigger_types[];


void TriggerBxMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>( "l1tResults", edm::InputTag("gtStage2Digis"));
  desc.addUntracked<edm::InputTag>( "hltResults", edm::InputTag("TriggerResults"));
  desc.addUntracked<std::string>(   "dqmPath",    "HLT/TriggerBx" );
  desc.addUntracked<bool>(          "make1DPlots", true);
  desc.addUntracked<bool>(          "make2DPlots", false);
  desc.addUntracked<uint32_t>(      "lsRange",     4000);
  descriptions.add("triggerBxMonitor", desc);
}


TriggerBxMonitor::TriggerBxMonitor(edm::ParameterSet const & config) :
  // module configuration
  m_l1t_results( consumes<GlobalAlgBlkBxCollection>( config.getUntrackedParameter<edm::InputTag>( "l1tResults" ) ) ),
  m_hlt_results( consumes<edm::TriggerResults>(      config.getUntrackedParameter<edm::InputTag>( "hltResults" ) ) ),
  m_dqm_path(                                        config.getUntrackedParameter<std::string>(   "dqmPath" ) ),
  m_make_1d_plots(                                   config.getUntrackedParameter<bool>(          "make1DPlots" ) ),
  m_make_2d_plots(                                   config.getUntrackedParameter<bool>(          "make2DPlots" ) ),
  m_ls_range(                                        config.getUntrackedParameter<uint32_t>(      "lsRange" ) ),
  // L1T and HLT configuration
  m_l1tMenu(nullptr),
  m_hltConfig(),
  // L1T and HLT results
  m_tcds_bx_all(nullptr),
  m_l1t_bx_all(nullptr),
  m_hlt_bx_all(nullptr),
  m_tcds_bx(),
  m_l1t_bx(),
  m_hlt_bx(),
  m_tcds_bx_2d(),
  m_l1t_bx_2d(),
  m_hlt_bx_2d()
{
}

TriggerBxMonitor::~TriggerBxMonitor()
{
}

void TriggerBxMonitor::dqmBeginRun(edm::Run const & run, edm::EventSetup const & setup)
{
  // initialise the TCDS vector
  if (m_make_1d_plots) {
    m_tcds_bx.clear();
    m_tcds_bx.resize(sizeof(s_tcds_trigger_types) / sizeof(const char *), nullptr);
  }
  if (m_make_2d_plots) {
    m_tcds_bx_2d.clear();
    m_tcds_bx_2d.resize(sizeof(s_tcds_trigger_types) / sizeof(const char *), nullptr);
  }

  // cache the L1 trigger menu
  m_l1tMenu = & get<L1TUtmTriggerMenuRcd, L1TUtmTriggerMenu>(setup);
  if (m_l1tMenu) {
    if (m_make_1d_plots) {
      m_l1t_bx.clear();
      m_l1t_bx.resize(GlobalAlgBlk::maxPhysicsTriggers, nullptr);
    }
    if (m_make_2d_plots) {
      m_l1t_bx_2d.clear();
      m_l1t_bx_2d.resize(GlobalAlgBlk::maxPhysicsTriggers, nullptr);
    }
  } else {
    edm::LogError("TriggerBxMonitor") << "failed to read the L1 menu from the EventSetup, the L1 trigger bx distribution will not be monitored";
  }

  // initialise the HLTConfigProvider
  bool changed = true;
  edm::EDConsumerBase::Labels labels;
  labelsForToken(m_hlt_results, labels);
  if (m_hltConfig.init(run, setup, labels.process, changed)) {
    if (m_make_1d_plots) {
      m_hlt_bx.clear();
      m_hlt_bx.resize( m_hltConfig.size(), nullptr );
    }
    if (m_make_2d_plots) {
      m_hlt_bx_2d.clear();
      m_hlt_bx_2d.resize( m_hltConfig.size(), nullptr );
    }
  } else {
    // HLTConfigProvider not initialised, skip the the HLT monitoring
    edm::LogError("TriggerBxMonitor") << "failed to initialise HLTConfigProvider, the HLT bx distribution will not be monitored";
  }
}

void TriggerBxMonitor::bookHistograms(DQMStore::IBooker & booker, edm::Run const & run, edm::EventSetup const & setup)
{
  // TCDS trigger type plots
  {
    size_t size = sizeof(s_tcds_trigger_types) / sizeof(const char *);

    // book 2D histogram to monitor all TCDS trigger types in a single plot
    booker.setCurrentFolder( m_dqm_path );
    m_tcds_bx_all = booker.book2D("TCDS Trigger Types", "TCDS Trigger Types vs. bunch crossing", s_bx_range + 1, -0.5, s_bx_range + 0.5, size, -0.5, size - 0.5)->getTH2F();

    // book the individual histograms for the known TCDS trigger types
    booker.setCurrentFolder( m_dqm_path + "/TCDS" );
    for (unsigned int i = 0; i < size; ++i) {
      if (s_tcds_trigger_types[i]) {
        if (m_make_1d_plots) {
          m_tcds_bx.at(i) = booker.book1D(s_tcds_trigger_types[i], s_tcds_trigger_types[i], s_bx_range + 1, -0.5, s_bx_range + 0.5)->getTH1F();
        }
        if (m_make_2d_plots) {
          std::string const & name_ls = std::string(s_tcds_trigger_types[i]) + " vs LS";
          m_tcds_bx_2d.at(i) = booker.book2D(name_ls, name_ls, s_bx_range + 1, -0.5, s_bx_range + 0.5, m_ls_range, 0.5, m_ls_range + 0.5)->getTH2F();
        }
        m_tcds_bx_all->GetYaxis()->SetBinLabel(i+1, s_tcds_trigger_types[i]);
      }
    }
  }

  // L1T plots
  if (m_l1tMenu) {
    // book 2D histogram to monitor all L1 triggers in a single plot
    booker.setCurrentFolder( m_dqm_path );
    m_l1t_bx_all = booker.book2D("Level 1 Triggers", "Level 1 Triggers vs. bunch crossing", s_bx_range + 1, -0.5, s_bx_range + 0.5, GlobalAlgBlk::maxPhysicsTriggers, -0.5, GlobalAlgBlk::maxPhysicsTriggers - 0.5)->getTH2F();

    // book the individual histograms for the L1 triggers that are included in the L1 menu
    booker.setCurrentFolder( m_dqm_path + "/L1T" );
    for (auto const & keyval: m_l1tMenu->getAlgorithmMap()) {
      unsigned int bit = keyval.second.getIndex();
      std::string const & name = (boost::format("%s (bit %d)") % keyval.first % bit).str();
      if (m_make_1d_plots) {
        m_l1t_bx.at(bit) = booker.book1D(name, name, s_bx_range + 1, -0.5, s_bx_range + 0.5)->getTH1F();
      }
      if (m_make_2d_plots) {
        std::string const & name_ls = name + " vs LS";
        m_l1t_bx_2d.at(bit) = booker.book2D(name_ls, name_ls, s_bx_range + 1, -0.5, s_bx_range + 0.5, m_ls_range, 0.5, m_ls_range + 0.5)->getTH2F();
      }
      m_l1t_bx_all->GetYaxis()->SetBinLabel(bit+1, keyval.first.c_str());
    }
  }

  // HLT plots
  if (m_hltConfig.inited()) {
    // book 2D histogram to monitor all HLT paths in a single plot
    booker.setCurrentFolder( m_dqm_path );
    m_hlt_bx_all = booker.book2D("High Level Triggers", "High Level Triggers vs. bunch crossing", s_bx_range + 1, -0.5, s_bx_range + 0.5, m_hltConfig.size(), -0.5, m_hltConfig.size() - 0.5)->getTH2F();

    // book the individual HLT triggers histograms
    booker.setCurrentFolder( m_dqm_path + "/HLT" );
    for (unsigned int i = 0; i < m_hltConfig.size(); ++i) {
      std::string const & name = m_hltConfig.triggerName(i);
      if (m_make_1d_plots) {
        m_hlt_bx[i] = booker.book1D(name, name, s_bx_range + 1, -0.5, s_bx_range + 0.5)->getTH1F();
      }
      if (m_make_2d_plots) {
        std::string const & name_ls = name + " vs LS";
        m_hlt_bx_2d[i] = booker.book2D(name_ls, name_ls, s_bx_range + 1, -0.5, s_bx_range + 0.5, m_ls_range, 0.5, m_ls_range + 0.5)->getTH2F();
      }
      m_hlt_bx_all->GetYaxis()->SetBinLabel(i+1, name.c_str());
    }
  }
}


void TriggerBxMonitor::analyze(edm::Event const & event, edm::EventSetup const & setup)
{
  unsigned int bx = event.bunchCrossing();
  unsigned int ls = event.luminosityBlock();

  // monitor the bx distribution for the TCDS trigger types
  {
    size_t size = sizeof(s_tcds_trigger_types) / sizeof(const char *);
    unsigned int type = event.experimentType();
    if (type < size) {
      if (m_make_1d_plots and m_tcds_bx.at(type))
        m_tcds_bx[type]->Fill(bx);
      if (m_make_2d_plots and m_tcds_bx_2d.at(type))
        m_tcds_bx_2d[type]->Fill(bx, ls);
    }
    m_tcds_bx_all->Fill(bx, type);
  }

  // monitor the bx distribution for the L1 triggers
  if (m_l1tMenu) {
    auto const & bxvector = get<GlobalAlgBlkBxCollection>(event, m_l1t_results);
    if (not bxvector.isEmpty(0)) {
      auto const & results = bxvector.at(0, 0);
      for (unsigned int i = 0; i < GlobalAlgBlk::maxPhysicsTriggers; ++i)
        if (results.getAlgoDecisionFinal(i)) {
          if (m_make_1d_plots and m_l1t_bx.at(i))
            m_l1t_bx[i]->Fill(bx);
          if (m_make_2d_plots and m_l1t_bx_2d.at(i))
            m_l1t_bx_2d[i]->Fill(bx, ls);
          m_l1t_bx_all->Fill(bx, i);
        }
    }
  }

  // monitor the bx distribution for the HLT triggers
  if (m_hltConfig.inited()) {
    auto const & hltResults = get<edm::TriggerResults>(event, m_hlt_results);
    for (unsigned int i = 0; i < hltResults.size(); ++i) {
      if (hltResults.at(i).accept()) {
        if (m_make_1d_plots and m_hlt_bx.at(i))
          m_hlt_bx[i]->Fill(bx);
        if (m_make_2d_plots and m_hlt_bx_2d.at(i))
          m_hlt_bx_2d[i]->Fill(bx, ls);
        m_hlt_bx_all->Fill(bx, i);
      }
    }
  }
}


//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerBxMonitor);
