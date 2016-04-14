#ifndef DQM_L1TMonitor_L1TStage2BMTF_h
#define DQM_L1TMonitor_L1TStage2BMTF_h


#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class L1TStage2BMTF : public DQMEDAnalyzer {

 public:

  L1TStage2BMTF(const edm::ParameterSet& ps);
  virtual ~L1TStage2BMTF();

 protected:

  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

 private:  

  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> bmtfToken;
  std::string monitorDir;
  bool verbose;

  MonitorElement* bmtfBX;
  MonitorElement* bmtfhwPt;
  MonitorElement* bmtfhwEta;
  MonitorElement* bmtfhwPhi;

  MonitorElement* bmtfhwPtvshwEta;
  MonitorElement* bmtfhwPtvshwPhi;
  MonitorElement* bmtfhwPhivshwEta;

  MonitorElement* bmtfBXvshwPt;
  MonitorElement* bmtfBXvshwEta;
  MonitorElement* bmtfBXvshwPhi;
};

#endif
