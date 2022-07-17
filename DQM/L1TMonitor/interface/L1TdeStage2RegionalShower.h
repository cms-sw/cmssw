#ifndef DQM_L1TMonitor_L1TdeStage2RegionalShower_h
#define DQM_L1TMonitor_L1TdeStage2RegionalShower_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"

class L1TdeStage2RegionalShower : public DQMEDAnalyzer {
public:
  L1TdeStage2RegionalShower(const edm::ParameterSet& ps);
  ~L1TdeStage2RegionalShower() override;

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<l1t::RegionalMuonShowerBxCollection> data_EMTFShower_token_;
  edm::EDGetTokenT<l1t::RegionalMuonShowerBxCollection> emul_EMTFShower_token_;

  std::string monitorDir_;

  MonitorElement* emtfShowerDataSummary_denom_;
  MonitorElement* emtfShowerDataSummary_num_;
  MonitorElement* emtfShowerEmulSummary_denom_;
  MonitorElement* emtfShowerEmulSummary_num_;
};

#endif
