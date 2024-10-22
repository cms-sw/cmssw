#ifndef DQM_L1TMonitor_L1TStage2uGMTInputBxDistributions_h
#define DQM_L1TMonitor_L1TStage2uGMTInputBxDistributions_h

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1Trigger/interface/MuonShower.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class L1TStage2uGMTInputBxDistributions : public DQMEDAnalyzer {
public:
  L1TStage2uGMTInputBxDistributions(const edm::ParameterSet& ps);
  ~L1TStage2uGMTInputBxDistributions() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> ugmtBMTFToken_;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> ugmtOMTFToken_;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> ugmtEMTFToken_;
  edm::EDGetTokenT<l1t::MuonBxCollection> ugmtMuonToken_;
  edm::EDGetTokenT<l1t::RegionalMuonShowerBxCollection> ugmtEMTFShowerToken_;
  edm::EDGetTokenT<l1t::MuonShowerBxCollection> ugmtMuonShowerToken_;
  std::string monitorDir_;
  bool emul_;
  bool verbose_;
  bool hadronicShowers_;

  MonitorElement* ugmtBMTFBX;

  MonitorElement* ugmtOMTFBX;

  MonitorElement* ugmtEMTFBX;

  MonitorElement* ugmtEMTFShowerTypeOccupancyPerBx;
  MonitorElement* ugmtEMTFShowerSectorOccupancyPerBx;

  MonitorElement* ugmtBMTFBXvsProcessor;
  MonitorElement* ugmtOMTFBXvsProcessor;
  MonitorElement* ugmtEMTFBXvsProcessor;
  MonitorElement* ugmtBXvsLink;

  static constexpr unsigned IDX_LOOSE_SHOWER{3};
  static constexpr unsigned IDX_TIGHT_SHOWER{2};
  static constexpr unsigned IDX_NOMINAL_SHOWER{1};
};

#endif
