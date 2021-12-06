#ifndef DQM_L1TMonitor_L1TdeCSCTPGShower_h
#define DQM_L1TMonitor_L1TdeCSCTPGShower_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/CSCDigi/interface/CSCShowerDigiCollection.h"

class L1TdeCSCTPGShower : public DQMEDAnalyzer {
public:
  L1TdeCSCTPGShower(const edm::ParameterSet& ps);
  ~L1TdeCSCTPGShower() override;

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  bool areSameShowers(const CSCShowerDigi& lhs, const CSCShowerDigi& rhs) const;

  edm::EDGetTokenT<CSCShowerDigiCollection> dataALCTShower_token_;
  edm::EDGetTokenT<CSCShowerDigiCollection> emulALCTShower_token_;
  edm::EDGetTokenT<CSCShowerDigiCollection> dataCLCTShower_token_;
  edm::EDGetTokenT<CSCShowerDigiCollection> emulCLCTShower_token_;
  edm::EDGetTokenT<CSCShowerDigiCollection> dataLCTShower_token_;
  edm::EDGetTokenT<CSCShowerDigiCollection> emulLCTShower_token_;

  std::string monitorDir_;

  MonitorElement* lctShowerDataSummary_denom_;
  MonitorElement* lctShowerDataSummary_num_;
  MonitorElement* alctShowerDataSummary_denom_;
  MonitorElement* alctShowerDataSummary_num_;
  MonitorElement* clctShowerDataSummary_denom_;
  MonitorElement* clctShowerDataSummary_num_;

  MonitorElement* lctShowerEmulSummary_denom_;
  MonitorElement* lctShowerEmulSummary_num_;
  MonitorElement* alctShowerEmulSummary_denom_;
  MonitorElement* alctShowerEmulSummary_num_;
  MonitorElement* clctShowerEmulSummary_denom_;
  MonitorElement* clctShowerEmulSummary_num_;
};

#endif
