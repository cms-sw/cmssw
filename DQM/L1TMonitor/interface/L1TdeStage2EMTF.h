#ifndef DQM_L1TMonitor_L1TdeStage2EMTF_h
#define DQM_L1TMonitor_L1TdeStage2EMTF_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

class L1TdeStage2EMTF : public DQMEDAnalyzer {
public:
  L1TdeStage2EMTF(const edm::ParameterSet& ps);
  ~L1TdeStage2EMTF() override;

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> dataToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> emulToken;
  std::string monitorDir;
  bool verbose;

  MonitorElement* emtfComparenMuonsEvent;

  MonitorElement* emtfDataBX;
  MonitorElement* emtfEmulBX;
  MonitorElement* emtfDatahwPt;
  MonitorElement* emtfEmulhwPt;
  MonitorElement* emtfDatahwEta;
  MonitorElement* emtfEmulhwEta;
  MonitorElement* emtfDatahwPhi;
  MonitorElement* emtfEmulhwPhi;
  MonitorElement* emtfDatahwQual;
  MonitorElement* emtfEmulhwQual;

  /*MonitorElement* emtfComparehwPt;
  MonitorElement* emtfComparehwEta;
  MonitorElement* emtfComparehwPhi;
  MonitorElement* emtfComparehwQual;*/
};

#endif
