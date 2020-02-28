#ifndef DQM_L1TMonitor_L1TStage2EMTF_h
#define DQM_L1TMonitor_L1TStage2EMTF_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

class L1TStage2EMTF : public DQMEDAnalyzer {
public:
  L1TStage2EMTF(const edm::ParameterSet& ps);
  ~L1TStage2EMTF() override;

protected:
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
  MonitorElement* cscDQMOccupancy;
  MonitorElement* cscLCTTiming[5];
  MonitorElement* cscLCTTimingFrac[5];
  MonitorElement* cscTimingTot;

  MonitorElement* emtfnTracks;
  MonitorElement* emtfTracknHits;
  MonitorElement* emtfTrackBX;
  MonitorElement* emtfTrackBXVsCSCLCT[3];
  MonitorElement* emtfTrackBXVsRPCHit[3];
  MonitorElement* emtfTrackPt;
  MonitorElement* emtfTrackPtHighQuality;  //Chad Freer May 8 2018
  MonitorElement* emtfTrackPtHighQualityHighPT;
  MonitorElement* emtfTrackEta;
  MonitorElement* emtfTrackEtaHighQuality;  //Chad Freer May 8 2018
  MonitorElement* emtfTrackEtaHighQualityHighPT;
  MonitorElement* emtfTrackPhi;
  MonitorElement* emtfTrackPhiHighQuality;
  MonitorElement* emtfTrackPhiHighQualityHighPT;
  MonitorElement* emtfTrackOccupancy;
  MonitorElement* emtfTrackOccupancyHighQuality;  //Chad Freer May 8 2018
  MonitorElement* emtfTrackOccupancyHighQualityHighPT;
  MonitorElement* emtfTrackMode;
  MonitorElement* emtfTrackQuality;
  MonitorElement* emtfTrackQualityVsMode;
  MonitorElement* RPCvsEMTFTrackMode;

  MonitorElement* emtfMuonBX;
  MonitorElement* emtfMuonhwPt;
  MonitorElement* emtfMuonhwEta;
  MonitorElement* emtfMuonhwPhi;
  MonitorElement* emtfMuonhwQual;

  MonitorElement* rpcHitBX;
  MonitorElement* rpcHitOccupancy;
  MonitorElement* rpcHitTiming[5];
  MonitorElement* rpcHitTimingFrac[5];
  MonitorElement* rpcHitTimingTot;
  MonitorElement* rpcHitPhi[12];
  MonitorElement* rpcHitTheta[12];
  MonitorElement* rpcChamberPhi[12];
  MonitorElement* rpcChamberTheta[12];

  MonitorElement* rpcHitTimingInTrack;
};

#endif
