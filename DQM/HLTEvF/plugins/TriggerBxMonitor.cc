// C++ headers
#include <cstring>
#include <iterator>
#include <string>

// boost headers
#include <boost/format.hpp>

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
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

namespace {

  struct RunBasedHistograms {
  public:
    typedef dqm::reco::MonitorElement MonitorElement;
    RunBasedHistograms()
        :  // L1T and HLT configuration
          hltConfig(),
          // L1T and HLT results
          tcds_bx_all(nullptr),
          l1t_bx_all(nullptr),
          hlt_bx_all(nullptr),
          tcds_bx(),
          l1t_bx(),
          hlt_bx(),
          tcds_bx_2d(),
          l1t_bx_2d(),
          hlt_bx_2d() {}

  public:
    // HLT configuration
    HLTConfigProvider hltConfig;

    // L1T and HLT results
    dqm::reco::MonitorElement* tcds_bx_all;
    dqm::reco::MonitorElement* l1t_bx_all;
    dqm::reco::MonitorElement* hlt_bx_all;
    std::vector<dqm::reco::MonitorElement*> tcds_bx;
    std::vector<dqm::reco::MonitorElement*> l1t_bx;
    std::vector<dqm::reco::MonitorElement*> hlt_bx;
    std::vector<dqm::reco::MonitorElement*> tcds_bx_2d;
    std::vector<dqm::reco::MonitorElement*> l1t_bx_2d;
    std::vector<dqm::reco::MonitorElement*> hlt_bx_2d;
  };

}  // namespace

class TriggerBxMonitor : public DQMGlobalEDAnalyzer<RunBasedHistograms> {
public:
  explicit TriggerBxMonitor(edm::ParameterSet const&);
  ~TriggerBxMonitor() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&, RunBasedHistograms&) const override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, RunBasedHistograms&) const override;
  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, RunBasedHistograms const&) const override;

  // number of bunch crossings
  static const unsigned int s_bx_range = 3564;

  // TCDS trigger types
  // see https://twiki.cern.ch/twiki/bin/viewauth/CMS/TcdsEventRecord
  static constexpr const char* s_tcds_trigger_types[] = {
      "Empty",          //  0 - No trigger
      "Physics",        //  1 - GT trigger
      "Calibration",    //  2 - Sequence trigger (calibration)
      "Random",         //  3 - Random trigger
      "Auxiliary",      //  4 - Auxiliary (CPM front panel NIM input) trigger
      nullptr,          //  5 - reserved
      nullptr,          //  6 - reserved
      nullptr,          //  7 - reserved
      "Cyclic",         //  8 - Cyclic trigger
      "Bunch-pattern",  //  9 - Bunch-pattern trigger
      "Software",       // 10 - Software trigger
      "TTS",            // 11 - TTS-sourced trigger
      nullptr,          // 12 - reserved
      nullptr,          // 13 - reserved
      nullptr,          // 14 - reserved
      nullptr           // 15 - reserved
  };

  // module configuration
  const edm::EDGetTokenT<GlobalAlgBlkBxCollection> m_l1t_results;
  const edm::EDGetTokenT<edm::TriggerResults> m_hlt_results;
  const std::string m_dqm_path;
  const bool m_make_1d_plots;
  const bool m_make_2d_plots;
  const uint32_t m_ls_range;
};

// definition
constexpr const char* TriggerBxMonitor::s_tcds_trigger_types[];

void TriggerBxMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("l1tResults", edm::InputTag("gtStage2Digis"));
  desc.addUntracked<edm::InputTag>("hltResults", edm::InputTag("TriggerResults"));
  desc.addUntracked<std::string>("dqmPath", "HLT/TriggerBx");
  desc.addUntracked<bool>("make1DPlots", true);
  desc.addUntracked<bool>("make2DPlots", false);
  desc.addUntracked<uint32_t>("lsRange", 4000);
  descriptions.add("triggerBxMonitor", desc);
}

TriggerBxMonitor::TriggerBxMonitor(edm::ParameterSet const& config)
    :  // module configuration
      m_l1t_results(consumes<GlobalAlgBlkBxCollection>(config.getUntrackedParameter<edm::InputTag>("l1tResults"))),
      m_hlt_results(consumes<edm::TriggerResults>(config.getUntrackedParameter<edm::InputTag>("hltResults"))),
      m_dqm_path(config.getUntrackedParameter<std::string>("dqmPath")),
      m_make_1d_plots(config.getUntrackedParameter<bool>("make1DPlots")),
      m_make_2d_plots(config.getUntrackedParameter<bool>("make2DPlots")),
      m_ls_range(config.getUntrackedParameter<uint32_t>("lsRange")) {}

void TriggerBxMonitor::dqmBeginRun(edm::Run const& run,
                                   edm::EventSetup const& setup,
                                   RunBasedHistograms& histograms) const {
  // initialise the TCDS vector
  if (m_make_1d_plots) {
    histograms.tcds_bx.clear();
    histograms.tcds_bx.resize(std::size(s_tcds_trigger_types));
  }
  if (m_make_2d_plots) {
    histograms.tcds_bx_2d.clear();
    histograms.tcds_bx_2d.resize(std::size(s_tcds_trigger_types));
  }

  // cache the L1 trigger menu
  if (m_make_1d_plots) {
    histograms.l1t_bx.clear();
    histograms.l1t_bx.resize(GlobalAlgBlk::maxPhysicsTriggers);
  }
  if (m_make_2d_plots) {
    histograms.l1t_bx_2d.clear();
    histograms.l1t_bx_2d.resize(GlobalAlgBlk::maxPhysicsTriggers);
  }

  // initialise the HLTConfigProvider
  bool changed = true;
  edm::EDConsumerBase::Labels labels;
  labelsForToken(m_hlt_results, labels);
  if (histograms.hltConfig.init(run, setup, labels.process, changed)) {
    if (m_make_1d_plots) {
      histograms.hlt_bx.clear();
      histograms.hlt_bx.resize(histograms.hltConfig.size());
    }
    if (m_make_2d_plots) {
      histograms.hlt_bx_2d.clear();
      histograms.hlt_bx_2d.resize(histograms.hltConfig.size());
    }
  } else {
    // HLTConfigProvider not initialised, skip the the HLT monitoring
    edm::LogError("TriggerBxMonitor")
        << "failed to initialise HLTConfigProvider, the HLT bx distribution will not be monitored";
  }
}

void TriggerBxMonitor::bookHistograms(DQMStore::IBooker& booker,
                                      edm::Run const& run,
                                      edm::EventSetup const& setup,
                                      RunBasedHistograms& histograms) const {
  // TCDS trigger type plots
  {
    size_t size = std::size(s_tcds_trigger_types);

    // book 2D histogram to monitor all TCDS trigger types in a single plot
    booker.setCurrentFolder(m_dqm_path);
    histograms.tcds_bx_all = booker.book2D("TCDS Trigger Types",
                                           "TCDS Trigger Types vs. bunch crossing",
                                           s_bx_range + 1,
                                           -0.5,
                                           s_bx_range + 0.5,
                                           size,
                                           -0.5,
                                           size - 0.5);

    // book the individual histograms for the known TCDS trigger types
    booker.setCurrentFolder(m_dqm_path + "/TCDS");
    for (unsigned int i = 0; i < size; ++i) {
      if (s_tcds_trigger_types[i]) {
        if (m_make_1d_plots) {
          histograms.tcds_bx.at(i) =
              booker.book1D(s_tcds_trigger_types[i], s_tcds_trigger_types[i], s_bx_range + 1, -0.5, s_bx_range + 0.5);
        }
        if (m_make_2d_plots) {
          std::string const& name_ls = std::string(s_tcds_trigger_types[i]) + " vs LS";
          histograms.tcds_bx_2d.at(i) = booker.book2D(
              name_ls, name_ls, s_bx_range + 1, -0.5, s_bx_range + 0.5, m_ls_range, 0.5, m_ls_range + 0.5);
        }
        histograms.tcds_bx_all->setBinLabel(i + 1, s_tcds_trigger_types[i], 2);  // Y axis
      }
    }
  }

  // L1T plots
  {
    // book 2D histogram to monitor all L1 triggers in a single plot
    booker.setCurrentFolder(m_dqm_path);
    histograms.l1t_bx_all = booker.book2D("Level 1 Triggers",
                                          "Level 1 Triggers vs. bunch crossing",
                                          s_bx_range + 1,
                                          -0.5,
                                          s_bx_range + 0.5,
                                          GlobalAlgBlk::maxPhysicsTriggers,
                                          -0.5,
                                          GlobalAlgBlk::maxPhysicsTriggers - 0.5);

    // book the individual histograms for the L1 triggers that are included in the L1 menu
    booker.setCurrentFolder(m_dqm_path + "/L1T");
    auto const& l1tMenu = edm::get<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd>(setup);
    for (auto const& keyval : l1tMenu.getAlgorithmMap()) {
      unsigned int bit = keyval.second.getIndex();
      std::string const& name = (boost::format("%s (bit %d)") % keyval.first % bit).str();
      if (m_make_1d_plots) {
        histograms.l1t_bx.at(bit) = booker.book1D(name, name, s_bx_range + 1, -0.5, s_bx_range + 0.5);
      }
      if (m_make_2d_plots) {
        std::string const& name_ls = name + " vs LS";
        histograms.l1t_bx_2d.at(bit) =
            booker.book2D(name_ls, name_ls, s_bx_range + 1, -0.5, s_bx_range + 0.5, m_ls_range, 0.5, m_ls_range + 0.5);
      }
      histograms.l1t_bx_all->setBinLabel(bit + 1, keyval.first, 2);  // Y axis
    }
  }

  // HLT plots
  if (histograms.hltConfig.inited()) {
    // book 2D histogram to monitor all HLT paths in a single plot
    booker.setCurrentFolder(m_dqm_path);
    histograms.hlt_bx_all = booker.book2D("High Level Triggers",
                                          "High Level Triggers vs. bunch crossing",
                                          s_bx_range + 1,
                                          -0.5,
                                          s_bx_range + 0.5,
                                          histograms.hltConfig.size(),
                                          -0.5,
                                          histograms.hltConfig.size() - 0.5);

    // book the individual HLT triggers histograms
    booker.setCurrentFolder(m_dqm_path + "/HLT");
    for (unsigned int i = 0; i < histograms.hltConfig.size(); ++i) {
      std::string const& name = histograms.hltConfig.triggerName(i);
      if (m_make_1d_plots) {
        histograms.hlt_bx[i] = booker.book1D(name, name, s_bx_range + 1, -0.5, s_bx_range + 0.5);
      }
      if (m_make_2d_plots) {
        std::string const& name_ls = name + " vs LS";
        histograms.hlt_bx_2d[i] =
            booker.book2D(name_ls, name_ls, s_bx_range + 1, -0.5, s_bx_range + 0.5, m_ls_range, 0.5, m_ls_range + 0.5);
      }
      histograms.hlt_bx_all->setBinLabel(i + 1, name, 2);  // Y axis
    }
  }
}

void TriggerBxMonitor::dqmAnalyze(edm::Event const& event,
                                  edm::EventSetup const& setup,
                                  RunBasedHistograms const& histograms) const {
  unsigned int bx = event.bunchCrossing();
  unsigned int ls = event.luminosityBlock();

  // monitor the bx distribution for the TCDS trigger types
  {
    size_t size = std::size(s_tcds_trigger_types);
    unsigned int type = event.experimentType();
    if (type < size) {
      if (m_make_1d_plots and histograms.tcds_bx.at(type))
        histograms.tcds_bx[type]->Fill(bx);
      if (m_make_2d_plots and histograms.tcds_bx_2d.at(type))
        histograms.tcds_bx_2d[type]->Fill(bx, ls);
    }
    histograms.tcds_bx_all->Fill(bx, type);
  }

  // monitor the bx distribution for the L1 triggers
  {
    auto const& bxvector = edm::get(event, m_l1t_results);
    if (not bxvector.isEmpty(0)) {
      auto const& results = bxvector.at(0, 0);
      for (unsigned int i = 0; i < GlobalAlgBlk::maxPhysicsTriggers; ++i)
        if (results.getAlgoDecisionFinal(i)) {
          if (m_make_1d_plots and histograms.l1t_bx.at(i))
            histograms.l1t_bx[i]->Fill(bx);
          if (m_make_2d_plots and histograms.l1t_bx_2d.at(i))
            histograms.l1t_bx_2d[i]->Fill(bx, ls);
          histograms.l1t_bx_all->Fill(bx, i);
        }
    }
  }

  // monitor the bx distribution for the HLT triggers
  if (histograms.hltConfig.inited()) {
    auto const& hltResults = edm::get(event, m_hlt_results);
    for (unsigned int i = 0; i < hltResults.size(); ++i) {
      if (hltResults.at(i).accept()) {
        if (m_make_1d_plots and histograms.hlt_bx.at(i))
          histograms.hlt_bx[i]->Fill(bx);
        if (m_make_2d_plots and histograms.hlt_bx_2d.at(i))
          histograms.hlt_bx_2d[i]->Fill(bx, ls);
        histograms.hlt_bx_all->Fill(bx, i);
      }
    }
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerBxMonitor);
