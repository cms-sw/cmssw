#ifndef DQM_L1TMonitor_L1TStage2EMTF_h
#define DQM_L1TMonitor_L1TStage2EMTF_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"


class L1TStage2EMTF : public DQMEDAnalyzer {

 public:

  L1TStage2EMTF(const edm::ParameterSet& ps);
  virtual ~L1TStage2EMTF();

 protected:

  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

 private:

  edm::EDGetTokenT<l1t::EMTFDaqOutCollection> daqToken;
  edm::EDGetTokenT<l1t::EMTFHitCollection> hitToken;
  edm::EDGetTokenT<l1t::EMTFTrackCollection> trackToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> muonToken;
  std::string monitorDir;
  bool verbose;

  MonitorElement* emtfErrors;
  MonitorElement* mpcLinkErrors;
  MonitorElement* mpcLinkGood;

  MonitorElement* cscLCTBX;
  MonitorElement* cscLCTStrip[20];
  MonitorElement* cscLCTWire[20];
  MonitorElement* cscChamberStrip[20];
  MonitorElement* cscChamberWire[20];
  MonitorElement* cscLCTOccupancy;
  MonitorElement* cscLCTTiming[5];
  MonitorElement* cscLCTTimingFrac[4];
  
  MonitorElement* emtfnTracks;
  MonitorElement* emtfTracknHits;
  MonitorElement* emtfTrackBX;
  MonitorElement* emtfTrackBXVsCSCLCT[3];
  MonitorElement* emtfTrackBXVsRPCHit[3];
  MonitorElement* emtfTrackPt;
  MonitorElement* emtfTrackEta;
  MonitorElement* emtfTrackPhi;
  MonitorElement* emtfTrackPhiHighQuality;
  MonitorElement* emtfTrackOccupancy;
  MonitorElement* emtfTrackMode;
  MonitorElement* emtfTrackQuality;
  MonitorElement* emtfTrackQualityVsMode;

  MonitorElement* emtfMuonBX;
  MonitorElement* emtfMuonhwPt;
  MonitorElement* emtfMuonhwEta;
  MonitorElement* emtfMuonhwPhi;
  MonitorElement* emtfMuonhwQual;

  MonitorElement* rpcHitBX;
  MonitorElement* rpcHitOccupancy;
  MonitorElement* rpcHitTiming[5];
  MonitorElement* rpcHitTimingFrac[4];
  MonitorElement* rpcHitPhi[12];
  MonitorElement* rpcHitTheta[12];
  MonitorElement* rpcChamberPhi[12];
  MonitorElement* rpcChamberTheta[12];
  
  MonitorElement* rpcHitTimingInTrack;
};

#endif
