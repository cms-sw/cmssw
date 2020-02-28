// C++ headers
#include <cstring>
#include <string>

// CMSSW headers
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

namespace {
  typedef dqm::reco::DQMStore DQMStore;
  typedef dqm::reco::MonitorElement MonitorElement;

  struct RunBasedHistograms {
    dqm::reco::MonitorElement* orbit_bx_all;
    std::vector<dqm::reco::MonitorElement*> orbit_bx;
    std::vector<dqm::reco::MonitorElement*> orbit_bx_all_byLS;
  };
}  // namespace

class TriggerBxVsOrbitMonitor : public DQMGlobalEDAnalyzer<RunBasedHistograms> {
public:
  explicit TriggerBxVsOrbitMonitor(edm::ParameterSet const&);
  ~TriggerBxVsOrbitMonitor() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&, RunBasedHistograms&) const override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, RunBasedHistograms&) const override;
  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, RunBasedHistograms const&) const override;

  // number of bunch crossings
  static const unsigned int s_bx_range = 3564;
  static const unsigned int s_orbit_range = 262144;  // 2**18 orbits in 1 LS

  // TCDS trigger types
  // see https://twiki.cern.ch/twiki/bin/viewauth/CMS/TcdsEventRecord
  static constexpr const char* const s_tcds_trigger_types[] = {
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
  const int m_minLS;
  const int m_maxLS;
  const int m_minBX;
  const int m_maxBX;
};

// definition
constexpr const char* const TriggerBxVsOrbitMonitor::s_tcds_trigger_types[];

void TriggerBxVsOrbitMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("l1tResults", edm::InputTag("gtStage2Digis"));
  desc.addUntracked<edm::InputTag>("hltResults", edm::InputTag("TriggerResults"));
  desc.addUntracked<std::string>("dqmPath", "HLT/TriggerBx");
  desc.addUntracked<int>("minLS", 134);
  desc.addUntracked<int>("maxLS", 136);
  desc.addUntracked<int>("minBX", 894);
  desc.addUntracked<int>("maxBX", 912);
  descriptions.add("triggerBxVsOrbitMonitor", desc);
}

TriggerBxVsOrbitMonitor::TriggerBxVsOrbitMonitor(edm::ParameterSet const& config)
    :  // module configuration
      m_l1t_results(consumes<GlobalAlgBlkBxCollection>(config.getUntrackedParameter<edm::InputTag>("l1tResults"))),
      m_hlt_results(consumes<edm::TriggerResults>(config.getUntrackedParameter<edm::InputTag>("hltResults"))),
      m_dqm_path(config.getUntrackedParameter<std::string>("dqmPath")),
      m_minLS(config.getUntrackedParameter<int>("minLS")),
      m_maxLS(config.getUntrackedParameter<int>("maxLS")),
      m_minBX(config.getUntrackedParameter<int>("minBX")),
      m_maxBX(config.getUntrackedParameter<int>("maxBX")) {}

void TriggerBxVsOrbitMonitor::dqmBeginRun(edm::Run const& run,
                                          edm::EventSetup const& setup,
                                          RunBasedHistograms& histograms) const {
  size_t nLS = m_maxLS - m_minLS + 1;

  histograms.orbit_bx_all_byLS.clear();
  histograms.orbit_bx_all_byLS.resize(nLS);

  histograms.orbit_bx.clear();
  histograms.orbit_bx.resize(std::size(s_tcds_trigger_types));
}

void TriggerBxVsOrbitMonitor::bookHistograms(DQMStore::IBooker& booker,
                                             edm::Run const& run,
                                             edm::EventSetup const& setup,
                                             RunBasedHistograms& histograms) const {
  // TCDS trigger type plots
  size_t size = std::size(s_tcds_trigger_types);
  size_t nLS = m_maxLS - m_minLS + 1;
  size_t nBX = m_maxBX - m_minBX + 1;

  // book 2D histogram to monitor all TCDS trigger types in a single plot
  booker.setCurrentFolder(m_dqm_path + "/orbitVsBX");
  histograms.orbit_bx_all = booker.book2D("OrbitVsBX",
                                          "Event orbits vs. bunch crossing",
                                          nBX,
                                          m_minBX - 0.5,
                                          m_maxBX + 0.5,
                                          s_orbit_range + 1,
                                          -0.5,
                                          s_orbit_range + 0.5);
  histograms.orbit_bx_all->setXTitle("BX");
  histograms.orbit_bx_all->setYTitle("orbit");

  for (unsigned int i = 0; i < nLS; ++i) {
    std::string iname = std::to_string(i);
    histograms.orbit_bx_all_byLS[i] = booker.book2D("OrbitVsBX_LS" + iname,
                                                    "Event orbits vs. bunch crossing, for lumisection " + iname,
                                                    nBX,
                                                    m_minBX - 0.5,
                                                    m_maxBX + 0.5,
                                                    s_orbit_range + 1,
                                                    -0.5,
                                                    s_orbit_range + 0.5);
    histograms.orbit_bx_all_byLS[i]->setXTitle("BX");
    histograms.orbit_bx_all_byLS[i]->setYTitle("orbit");
  }

  booker.setCurrentFolder(m_dqm_path + "/orbitVsBX/TCDS");
  for (unsigned int i = 0; i < size; ++i) {
    if (s_tcds_trigger_types[i]) {
      histograms.orbit_bx[i] = booker.book2D("OrbitVsBX_" + std::string(s_tcds_trigger_types[i]),
                                             "Event orbits vs. bunch crossing " + std::string(s_tcds_trigger_types[i]),
                                             nBX,
                                             m_minBX - 0.5,
                                             m_maxBX + 0.5,
                                             s_orbit_range + 1,
                                             -0.5,
                                             s_orbit_range + 0.5);
      histograms.orbit_bx[i]->setXTitle("BX");
      histograms.orbit_bx[i]->setYTitle("orbit");
    }
  }
}

void TriggerBxVsOrbitMonitor::dqmAnalyze(edm::Event const& event,
                                         edm::EventSetup const& setup,
                                         RunBasedHistograms const& histograms) const {
  unsigned int type = event.experimentType();
  unsigned int ls = event.id().luminosityBlock();
  unsigned int orbit = event.orbitNumber() % s_orbit_range;
  unsigned int bx = event.bunchCrossing();
  histograms.orbit_bx_all->Fill(bx, orbit);

  int iLS = ls - m_minLS;
  if (iLS >= 0 and iLS < int(histograms.orbit_bx_all_byLS.size()))
    histograms.orbit_bx_all_byLS[iLS]->Fill(bx, orbit);

  // monitor the bx distribution for the TCDS trigger types
  size_t size = std::size(s_tcds_trigger_types);
  if (type < size and histograms.orbit_bx[type]) {
    histograms.orbit_bx[type]->Fill(bx, orbit);
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerBxVsOrbitMonitor);
