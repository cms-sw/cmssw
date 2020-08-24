#ifndef DQM_L1TMonitor_L1TdeCSCTPG_h
#define DQM_L1TMonitor_L1TdeCSCTPG_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

class L1TdeCSCTPG : public DQMEDAnalyzer {
public:
  L1TdeCSCTPG(const edm::ParameterSet& ps);
  ~L1TdeCSCTPG() override;

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<CSCALCTDigiCollection> dataALCT_token_;
  edm::EDGetTokenT<CSCALCTDigiCollection> emulALCT_token_;
  edm::EDGetTokenT<CSCCLCTDigiCollection> dataCLCT_token_;
  edm::EDGetTokenT<CSCCLCTDigiCollection> emulCLCT_token_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> dataLCT_token_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> emulLCT_token_;
  std::string monitorDir_;
  bool verbose_;

  // ALCT
  MonitorElement* alct_quality_emul_[10];
  MonitorElement* alct_wiregroup_emul_[10];
  MonitorElement* alct_bx_emul_[10];
  // ALCT
  MonitorElement* alct_quality_data_[10];
  MonitorElement* alct_wiregroup_data_[10];
  MonitorElement* alct_bx_data_[10];

  // CLCT
  MonitorElement* clct_pattern_emul_[10];
  MonitorElement* clct_quality_emul_[10];
  MonitorElement* clct_halfstrip_emul_[10];
  MonitorElement* clct_bend_emul_[10];
  MonitorElement* clct_bx_emul_[10];
  // CLCT
  MonitorElement* clct_pattern_data_[10];
  MonitorElement* clct_quality_data_[10];
  MonitorElement* clct_halfstrip_data_[10];
  MonitorElement* clct_bend_data_[10];
  MonitorElement* clct_bx_data_[10];

  // LCT
  MonitorElement* lct_pattern_emul_[10];
  MonitorElement* lct_quality_emul_[10];
  MonitorElement* lct_wiregroup_emul_[10];
  MonitorElement* lct_halfstrip_emul_[10];
  MonitorElement* lct_bend_emul_[10];
  MonitorElement* lct_bx_emul_[10];
  // LCT
  MonitorElement* lct_pattern_data_[10];
  MonitorElement* lct_quality_data_[10];
  MonitorElement* lct_wiregroup_data_[10];
  MonitorElement* lct_halfstrip_data_[10];
  MonitorElement* lct_bend_data_[10];
  MonitorElement* lct_bx_data_[10];
};

#endif
