#ifndef DQM_L1TMonitor_L1TStage2EMTF_h
#define DQM_L1TMonitor_L1TStage2EMTF_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/L1TMuon/interface/EMTFOutput.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"


class L1TStage2EMTF : public DQMEDAnalyzer {

 public:

  L1TStage2EMTF(const edm::ParameterSet& ps);
  virtual ~L1TStage2EMTF();

 protected:

  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

 private:

  edm::EDGetTokenT<l1t::EMTFOutputCollection> inputToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> outputToken;
  std::string monitorDir;
  bool verbose;

  MonitorElement* emtfErrors;
  MonitorElement* emtfLCTBX;
  MonitorElement* emtfLCTStrip[18];
  MonitorElement* emtfLCTWire[18];
  MonitorElement* emtfChamberStrip[18];
  MonitorElement* emtfChamberWire[18];
  MonitorElement* emtfChamberOccupancy;
  
  MonitorElement* emtfnTracksEvent;
  MonitorElement* emtfnTracksSP;
  MonitorElement* emtfnLCTs;
  MonitorElement* emtfTrackBX;
  MonitorElement* emtfTrackPt;
  MonitorElement* emtfTrackEta;
  MonitorElement* emtfTrackPhi;
  MonitorElement* emtfTrackOccupancy;
  MonitorElement* emtfMode;
  MonitorElement* emtfQuality;
  MonitorElement* emtfQualityvsMode;
  MonitorElement* emtfHQPhi;

  MonitorElement* emtfMuonBX;
  MonitorElement* emtfMuonhwPt;
  MonitorElement* emtfMuonhwEta;
  MonitorElement* emtfMuonhwPhi;
  MonitorElement* emtfMuonhwQual;
};

#endif
