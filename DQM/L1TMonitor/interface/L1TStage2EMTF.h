#ifndef DQM_L1TMonitor_L1TStage2EMTF_h
#define DQM_L1TMonitor_L1TStage2EMTF_h

#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


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

  edm::EDGetTokenT<l1t::EMTFDaqOutCollection> emtfToken;
  std::string monitorDir;
  bool verbose;

  MonitorElement* emtferrors;
  MonitorElement* emtflcts;
  MonitorElement* emtfChamberOccupancy;

  MonitorElement* emtf_strip_ME11_NEG;
  MonitorElement* emtf_wire_ME11_NEG;
  MonitorElement* emtf_strip_ME12_NEG;
  MonitorElement* emtf_wire_ME12_NEG;
  MonitorElement* emtf_strip_ME13_NEG;
  MonitorElement* emtf_wire_ME13_NEG;
  MonitorElement* emtf_strip_ME21_NEG;
  MonitorElement* emtf_wire_ME21_NEG;
  MonitorElement* emtf_strip_ME22_NEG;
  MonitorElement* emtf_wire_ME22_NEG;
  MonitorElement* emtf_strip_ME31_NEG;
  MonitorElement* emtf_wire_ME31_NEG;
  MonitorElement* emtf_strip_ME32_NEG;
  MonitorElement* emtf_wire_ME32_NEG;
  MonitorElement* emtf_strip_ME41_NEG;
  MonitorElement* emtf_wire_ME41_NEG;
  MonitorElement* emtf_strip_ME42_NEG;
  MonitorElement* emtf_wire_ME42_NEG;
  
  MonitorElement* emtf_strip_ME11_POS;
  MonitorElement* emtf_wire_ME11_POS;
  MonitorElement* emtf_strip_ME12_POS;
  MonitorElement* emtf_wire_ME12_POS;
  MonitorElement* emtf_strip_ME13_POS;
  MonitorElement* emtf_wire_ME13_POS;
  MonitorElement* emtf_strip_ME21_POS;
  MonitorElement* emtf_wire_ME21_POS;
  MonitorElement* emtf_strip_ME22_POS;
  MonitorElement* emtf_wire_ME22_POS;
  MonitorElement* emtf_strip_ME31_POS;
  MonitorElement* emtf_wire_ME31_POS;
  MonitorElement* emtf_strip_ME32_POS;
  MonitorElement* emtf_wire_ME32_POS;
  MonitorElement* emtf_strip_ME41_POS;
  MonitorElement* emtf_wire_ME41_POS;
  MonitorElement* emtf_strip_ME42_POS;
  MonitorElement* emtf_wire_ME42_POS;
  
  MonitorElement* emtf_chamberstrip_ME21_NEG;
  MonitorElement* emtf_chamberstrip_ME22_NEG;
  MonitorElement* emtf_chamberstrip_ME31_NEG;
  MonitorElement* emtf_chamberstrip_ME32_NEG;
  MonitorElement* emtf_chamberstrip_ME41_NEG;
  MonitorElement* emtf_chamberstrip_ME42_NEG;
  
  MonitorElement* emtf_chamberstrip_ME21_POS;
  MonitorElement* emtf_chamberstrip_ME22_POS;
  MonitorElement* emtf_chamberstrip_ME31_POS;
  MonitorElement* emtf_chamberstrip_ME32_POS;
  MonitorElement* emtf_chamberstrip_ME41_POS;
  MonitorElement* emtf_chamberstrip_ME42_POS;

  MonitorElement* emtfnTracks;
  MonitorElement* emtfnLCTs;
  MonitorElement* emtfTrackBX;
  MonitorElement* emtfTrackPt;
  MonitorElement* emtfTrackEta;
  MonitorElement* emtfTrackPhi;
  MonitorElement* emtfTrackOccupancy;
};

#endif
