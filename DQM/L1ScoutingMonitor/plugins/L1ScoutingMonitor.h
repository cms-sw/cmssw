#ifndef DQM_L1ScoutingMonitor_L1ScoutingMonitor_h
#define DQM_L1ScoutingMonitor_L1ScoutingMonitor_h

// system include files
#include <memory>
#include <iomanip>
#include <string>
#include <cmath>
#include <fstream>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// L1 Scouting
#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "L1TriggerScouting/Utilities/interface/conversion.h"
#include "L1TriggerScouting/Utilities/interface/convertToL1TFormat.h"

// L1Trigger
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

using namespace l1ScoutingRun3;

struct Histograms {
  typedef dqm::reco::MonitorElement MonitorElement;
  MonitorElement* histo_muon_BXocc_;
  MonitorElement* histo_jet_BXocc_;
  MonitorElement* histo_egamma_BXocc_;
  MonitorElement* histo_tau_BXocc_;
};

class L1ScoutingMonitor : public DQMGlobalEDAnalyzer<Histograms> {
public:
  explicit L1ScoutingMonitor(const edm::ParameterSet&);
  ~L1ScoutingMonitor() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  static const unsigned int s_bx_range = 3564;

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, Histograms&) const override;
  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms const&) const override;

  edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::Muon>> muonsTokenData_;
  edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::Jet>> jetsTokenData_;
  edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::EGamma>> eGammasTokenData_;
  edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::Tau>> tausTokenData_;

  const std::string m_dqm_path;
};

#endif
