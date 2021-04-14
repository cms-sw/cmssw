#ifndef DQM_L1TMonitor_L1TdeGEMTPG_h
#define DQM_L1TMonitor_L1TdeGEMTPG_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"

class L1TdeGEMTPG : public DQMEDAnalyzer {
public:
  L1TdeGEMTPG(const edm::ParameterSet& ps);
  ~L1TdeGEMTPG() override;

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<GEMPadDigiClusterCollection> data_token_;
  edm::EDGetTokenT<GEMPadDigiClusterCollection> emul_token_;
  std::string monitorDir_;
  bool verbose_;

  std::vector<std::string> chambers_;
  std::vector<std::string> dataEmul_;

  std::vector<std::string> clusterVars_;
  std::vector<unsigned> clusterNBin_;
  std::vector<double> clusterMinBin_;
  std::vector<double> clusterMaxBin_;

  // first key is the chamber number
  // second key is the variable
  std::map<uint32_t, std::map<std::string, MonitorElement*> > chamberHistos;
};

#endif
