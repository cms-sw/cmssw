#ifndef DQM_L1TMonitor_L1TdeCSCTPG_h
#define DQM_L1TMonitor_L1TdeCSCTPG_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

class L1TdeCSCTPG : public DQMEDAnalyzer {
public:
  L1TdeCSCTPG(const edm::ParameterSet& ps);
  ~L1TdeCSCTPG() override;

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // customized equality function
  bool areSameCLCTs(const CSCCLCTDigi& lhs, const CSCCLCTDigi& rhs) const;
  bool areSameLCTs(const CSCCorrelatedLCTDigi& lhs, const CSCCorrelatedLCTDigi& rhs) const;

  edm::EDGetTokenT<CSCALCTDigiCollection> dataALCT_token_;
  edm::EDGetTokenT<CSCALCTDigiCollection> emulALCT_token_;
  edm::EDGetTokenT<CSCCLCTDigiCollection> dataCLCT_token_;
  edm::EDGetTokenT<CSCCLCTDigiCollection> emulCLCT_token_;
  edm::EDGetTokenT<CSCCLCTPreTriggerDigiCollection> emulpreCLCT_token_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> dataLCT_token_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> emulLCT_token_;
  std::string monitorDir_;

  // ME1/1 combines trigger data from ME1/a and ME1/b
  std::vector<std::string> chambers_;
  std::vector<std::string> dataEmul_;

  std::vector<std::string> alctVars_;
  std::vector<std::string> clctVars_;
  std::vector<std::string> lctVars_;

  std::vector<unsigned> alctNBin_;
  std::vector<unsigned> clctNBin_;
  std::vector<unsigned> lctNBin_;
  std::vector<double> alctMinBin_;
  std::vector<double> clctMinBin_;
  std::vector<double> lctMinBin_;
  std::vector<double> alctMaxBin_;
  std::vector<double> clctMaxBin_;
  std::vector<double> lctMaxBin_;

  /*
    When set to True, we assume that the data comes from
    the Building 904 CSC test-stand. This test-stand is a single
    ME1/1 chamber, ME2/1, or ME4/2 chamber.
  */
  bool useB904_;
  bool useB904ME11_;
  bool useB904ME21_;
  bool useB904ME234s2_;

  bool isRun3_;
  /*
     By default the DQM will make 2D summary plots. Do you also want
     the very large number of 1D plots? Would recommend to keep it to
     true so that it may help in the debugging process (S.D.)
  */
  bool make1DPlots_;

  // check the data CLCTs and emul CLCTs against emul preCLCTs
  bool preTriggerAnalysis_;

  // first key is the chamber number
  // second key is the variable
  std::map<uint32_t, std::map<std::string, MonitorElement*> > chamberHistos;

  // 2D plots
  MonitorElement* lctDataSummary_denom_;
  MonitorElement* lctDataSummary_num_;
  MonitorElement* alctDataSummary_denom_;
  MonitorElement* alctDataSummary_num_;
  MonitorElement* clctDataSummary_denom_;
  MonitorElement* clctDataSummary_num_;

  MonitorElement* lctEmulSummary_denom_;
  MonitorElement* lctEmulSummary_num_;
  MonitorElement* alctEmulSummary_denom_;
  MonitorElement* alctEmulSummary_num_;
  MonitorElement* clctEmulSummary_denom_;
  MonitorElement* clctEmulSummary_num_;
};

#endif
