#ifndef DQM_L1TMonitor_L1TStage2EMTF_h
#define DQM_L1TMonitor_L1TStage2EMTF_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

class L1TStage2EMTF : public DQMOneEDAnalyzer<> {
public:
  L1TStage2EMTF(const edm::ParameterSet& ps);
  ~L1TStage2EMTF() override = default;

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
  MonitorElement* emtfTrackModeVsCSCBXDiff[8];  // Add mode vs BXdiff comparison Dec 07 2020

  MonitorElement* emtfnTracks;
  MonitorElement* emtfTracknHits;
  MonitorElement* emtfTrackBX;
  MonitorElement* emtfTrackBXVsCSCLCT[3];
  MonitorElement* emtfTrackBXVsRPCHit[3];
  MonitorElement* emtfTrackPt;
  MonitorElement* emtfTrackUnconstrainedPt;             // Lucas Faria de Sa Tucker Jun 28 2023
  MonitorElement* emtfTrackDxy;                         // Lucas Faria de Sa Tucker Jun 28 2023
  MonitorElement* emtfTrackPtHighQuality;               //Chad Freer May 8 2018
  MonitorElement* emtfTrackUnconstrainedPtHighQuality;  // Lucas Faria de Sa Tucker Jun 28 2023
  MonitorElement* emtfTrackPtHighQualityHighPT;
  MonitorElement* emtfTrackUnconstrainedPtHighQualityHighUPT;  // Lucas Faria de Sa Tucker Jun 28 2023
  MonitorElement* emtfTrackEta;
  MonitorElement* emtfTrackEtaHighQuality;  //Chad Freer May 8 2018
  MonitorElement* emtfTrackEtaHighQualityHighPT;
  MonitorElement* emtfTrackEtaHighQualityHighUPT;  // Lucas Faria de Sa Tucker Jun 28 2023
  MonitorElement* emtfTrackPhi;
  MonitorElement* emtfTrackPhiHighQuality;
  MonitorElement* emtfTrackPhiHighQualityHighPT;
  MonitorElement* emtfTrackPhiHighQualityHighUPT;  // Lucas Faria de Sa Tucker Jun 28 2023
  MonitorElement* emtfTrackOccupancy;
  MonitorElement* emtfTrackOccupancyHighQuality;  //Chad Freer May 8 2018
  MonitorElement* emtfTrackOccupancyHighQualityHighPT;
  MonitorElement* emtfTrackOccupancyHighQualityHighUPT;  // Lucas Faria de Sa Tucker Jun 28 2023
  MonitorElement* emtfTrackMode;
  MonitorElement* emtfTrackQuality;
  MonitorElement* emtfTrackQualityVsMode;
  MonitorElement* RPCvsEMTFTrackMode;

  MonitorElement* emtfMuonBX;
  MonitorElement* emtfMuonhwPt;
  MonitorElement* emtfMuonhwPtUnconstrained;  // Lucas Faria de Sa Tucker Jun 28 2023
  MonitorElement* emtfMuonhwDxy;              // Lucas Faria de Sa Tucker Jun 28 2023
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
  MonitorElement* emtfTrackModeVsRPCBXDiff[8];  // Add mode vs BXdiff comparison Dec 07 2020

  // Add GEMs Oct 27 2020
  MonitorElement* hitTypeBX;
  MonitorElement* gemHitBX;
  MonitorElement* gemHitOccupancy;
  MonitorElement* gemHitTiming[5];
  MonitorElement* gemHitTimingFrac[5];
  MonitorElement* gemHitTimingTot;
  MonitorElement* gemChamberPad[2];
  MonitorElement* gemChamberPartition[2];
  MonitorElement* emtfTrackBXVsGEMHit[3];
  MonitorElement* emtfTrackModeVsGEMBXDiff[2];  // Add mode vs BXdiff comparison Dec 07 2020

  // GEM vs CSC Dec 06 2020
  MonitorElement* gemHitPhi[2];
  MonitorElement* gemHitTheta[2];
  MonitorElement* gemHitVScscLCTPhi[2];
  MonitorElement* gemHitVScscLCTTheta[2];
  MonitorElement* gemHitVScscLCTBX[2];

  // GEM plots added July 21 2022
  MonitorElement* gemVFATBXPerChamber[36][2][2];
  MonitorElement* gemChamberVFATBX[2][7];
};

#endif
