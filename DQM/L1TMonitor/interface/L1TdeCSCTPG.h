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

  // first key is the chamber number
  // second key is the variable
  std::map<uint32_t, std::map<std::string, MonitorElement*> > chamberHistos;
};

#endif
