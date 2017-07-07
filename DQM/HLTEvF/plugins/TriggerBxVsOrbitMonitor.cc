// C++ headers
#include <string>
#include <cstring>

// boost headers
#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

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


class TriggerBxVsOrbitMonitor : public DQMEDAnalyzer {
public:
  explicit TriggerBxVsOrbitMonitor(edm::ParameterSet const &);
  ~TriggerBxVsOrbitMonitor() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  virtual void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(edm::Event const &, edm::EventSetup const &) override;

  // number of bunch crossings
  static const unsigned int s_bx_range = 3564;
  static const unsigned int s_orbit_range = 262144; // 2**18 orbits in 1 LS

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
  const int                                         m_minLS;
  const int                                         m_maxLS;
  const int                                         m_minBX;
  const int                                         m_maxBX;

  // L1T and HLT configuration
  L1TUtmTriggerMenu const * m_l1tMenu;
  HLTConfigProvider         m_hltConfig;

  std::vector<TH2F *>       m_orbit_bx_all_byLS;
  TH2F *                    m_orbit_bx_all;
  std::vector<TH2F *>       m_orbit_bx;
};

// definition
constexpr const char * const TriggerBxVsOrbitMonitor::s_tcds_trigger_types[];


void TriggerBxVsOrbitMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>( "l1tResults", edm::InputTag("gtStage2Digis"));
  desc.addUntracked<edm::InputTag>( "hltResults", edm::InputTag("TriggerResults"));
  desc.addUntracked<std::string>(   "dqmPath",    "HLT/TriggerBx" );
  desc.addUntracked<int>( "minLS", 134 );
  desc.addUntracked<int>( "maxLS", 136 );
  desc.addUntracked<int>( "minBX", 894 );
  desc.addUntracked<int>( "maxBX", 912 );
  descriptions.add("triggerBxVsOrbitMonitor", desc);
}

TriggerBxVsOrbitMonitor::TriggerBxVsOrbitMonitor(edm::ParameterSet const & config) :
  // module configuration
  m_l1t_results( consumes<GlobalAlgBlkBxCollection>( config.getUntrackedParameter<edm::InputTag>( "l1tResults" ) ) ),
  m_hlt_results( consumes<edm::TriggerResults>(      config.getUntrackedParameter<edm::InputTag>( "hltResults" ) ) ),
  m_dqm_path(                                        config.getUntrackedParameter<std::string>(   "dqmPath" ) ),
  m_minLS(                                           config.getUntrackedParameter<int>(           "minLS" ) ),
  m_maxLS(                                           config.getUntrackedParameter<int>(           "maxLS" ) ),
  m_minBX(                                           config.getUntrackedParameter<int>(           "minBX" ) ),
  m_maxBX(                                           config.getUntrackedParameter<int>(           "maxBX" ) ),
  // L1T and HLT configuration
  m_l1tMenu(nullptr),
  m_hltConfig(),
  m_orbit_bx_all_byLS(),
  m_orbit_bx_all(nullptr),
  m_orbit_bx()
{
}

void TriggerBxVsOrbitMonitor::dqmBeginRun(edm::Run const & run, edm::EventSetup const & setup)
{
  size_t nLS = m_maxLS-m_minLS+1;

  m_orbit_bx_all_byLS.clear();
  m_orbit_bx_all_byLS.resize(nLS,nullptr);

  m_orbit_bx.clear();
  m_orbit_bx.resize(sizeof(s_tcds_trigger_types) / sizeof(const char *), nullptr);
  

}

void TriggerBxVsOrbitMonitor::bookHistograms(DQMStore::IBooker & booker, edm::Run const & run, edm::EventSetup const & setup)
{
  // TCDS trigger type plots
  {
    size_t size = sizeof(s_tcds_trigger_types) / sizeof(const char *);
    size_t nLS = m_maxLS-m_minLS+1;

    unsigned int nBX = m_maxBX-m_minBX+1;
    // book 2D histogram to monitor all TCDS trigger types in a single plot
    booker.setCurrentFolder( m_dqm_path + "/orbitVsBX" );
    m_orbit_bx_all = booker.book2D("OrbitVsBX", "Event orbits vs. bunch crossing", nBX, float(m_minBX)-0.5, float(m_maxBX)+0.5, s_orbit_range+1, -0.5, float(s_orbit_range)+0.5)->getTH2F();
    m_orbit_bx_all->GetXaxis()->SetTitle("BX");
    m_orbit_bx_all->GetYaxis()->SetTitle("orbit");
    m_orbit_bx_all->SetCanExtend(TH1::kAllAxes);
    
    for (unsigned int i = 0; i < nLS; ++i) {
      std::string iname = std::to_string(i);
      m_orbit_bx_all_byLS.at(i) = booker.book2D("OrbitVsBX_LS"+iname, "OrbitVsBX_LS"+iname, nBX, float(m_minBX)-0.5, float(m_maxBX)+0.5, s_orbit_range+1, -0.5, float(s_orbit_range)+0.5)->getTH2F();
      m_orbit_bx_all_byLS.at(i)->GetXaxis()->SetTitle("BX");
      m_orbit_bx_all_byLS.at(i)->GetYaxis()->SetTitle("orbit");
    }
    
    booker.setCurrentFolder( m_dqm_path + "/orbitVsBX/TCDS" );
    for (unsigned int i = 0; i < size; ++i) {
      if (s_tcds_trigger_types[i]) {
	m_orbit_bx.at(i) = booker.book2D("OrbitVsBX_"+std::string(s_tcds_trigger_types[i]), "Event orbits vs. bunch crossing "+std::string(s_tcds_trigger_types[i]), nBX, float(m_minBX)-0.5, float(m_maxBX)+0.5,s_orbit_range+1, -0.5, float(s_orbit_range)+0.5)->getTH2F();
	m_orbit_bx.at(i)->GetXaxis()->SetTitle("BX");
	m_orbit_bx.at(i)->GetYaxis()->SetTitle("orbit");
      }
    }
  }
}


void TriggerBxVsOrbitMonitor::analyze(edm::Event const & event, edm::EventSetup const & setup)
{
  unsigned int bx    = event.bunchCrossing();
  unsigned int orbit = event.orbitNumber();
  unsigned int ls    = event.id().luminosityBlock();
  int orbit_in_ls = orbit-(s_orbit_range*(ls-1));
  m_orbit_bx_all->Fill(bx,orbit_in_ls);

  int iLS = ls-m_minLS;
  if (iLS >= 0 && iLS < int(m_orbit_bx_all_byLS.size()))
    m_orbit_bx_all_byLS.at(iLS)->Fill(bx,orbit_in_ls);
 

  // monitor the bx distribution for the TCDS trigger types
  size_t size = sizeof(s_tcds_trigger_types) / sizeof(const char *);
  unsigned int type = event.experimentType();
  if (type < size and m_orbit_bx[type]) {
    m_orbit_bx[type]->Fill(bx,orbit_in_ls);
  }

}


//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerBxVsOrbitMonitor);
