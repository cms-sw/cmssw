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

  MonitorElement* lctShowerDataNomSummary_denom_;
  MonitorElement* lctShowerDataNomSummary_num_;
  MonitorElement* alctShowerDataNomSummary_denom_;
  MonitorElement* alctShowerDataNomSummary_num_;
  MonitorElement* clctShowerDataNomSummary_denom_;
  MonitorElement* clctShowerDataNomSummary_num_;

  MonitorElement* lctShowerEmulNomSummary_denom_;
  MonitorElement* lctShowerEmulNomSummary_num_;
  MonitorElement* alctShowerEmulNomSummary_denom_;
  MonitorElement* alctShowerEmulNomSummary_num_;
  MonitorElement* clctShowerEmulNomSummary_denom_;
  MonitorElement* clctShowerEmulNomSummary_num_;

  MonitorElement* lctShowerDataTightSummary_denom_;
  MonitorElement* lctShowerDataTightSummary_num_;
  MonitorElement* alctShowerDataTightSummary_denom_;
  MonitorElement* alctShowerDataTightSummary_num_;
  MonitorElement* clctShowerDataTightSummary_denom_;
  MonitorElement* clctShowerDataTightSummary_num_;

  MonitorElement* lctShowerEmulTightSummary_denom_;
  MonitorElement* lctShowerEmulTightSummary_num_;
  MonitorElement* alctShowerEmulTightSummary_denom_;
  MonitorElement* alctShowerEmulTightSummary_num_;
  MonitorElement* clctShowerEmulTightSummary_denom_;
  MonitorElement* clctShowerEmulTightSummary_num_;
};

#endif
