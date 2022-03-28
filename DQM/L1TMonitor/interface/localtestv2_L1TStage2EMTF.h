#ifndef DQM_L1TMonitor_locv2_L1TStage2EMTF_h
#define DQM_L1TMonitor_locv2_L1TStage2EMTF_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

class locv2_L1TStage2EMTF : public DQMOneEDAnalyzer<> {
public:
  locv2_L1TStage2EMTF(const edm::ParameterSet& ps);
  ~locv2_L1TStage2EMTF() override;

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
  MonitorElement* emtfTrackModeVsCSCBXDiff[8]; // Add mode vs BXdiff comparison 2020.12.07

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
  MonitorElement* emtfTrackModeVsRPCBXDiff[6]; // Add mode vs BXdiff comparison 2020.12.07

  // Add GEMs Oct 27 2020
  MonitorElement* hitType;
  MonitorElement* hitTypeBX;
  MonitorElement* hitTypeSector;
  MonitorElement* hitTypeNumber; // GEM cosmics debug 2021.05.21
  MonitorElement* hitTypeNumSecGE11Pos; // GEM cosmics debug 2021.05.21
  MonitorElement* hitTypeNumSecGE11Neg; // GEM cosmics debug 2021.05.21
  MonitorElement* hitCoincideME11;      // GEM cosmics debug 2021.05.21
  MonitorElement* hitCoincideGE11;      // GEM cosmics debug 2021.05.21
  MonitorElement* SameSectorTimingCSCGEM; // GEM cosmics debug 2021.05.21
  MonitorElement* SameSectorChamberCSCGEM;   // GEM cosmics debug 2021.05.21
  MonitorElement* SameSectorGEMPadPartition; // GEM cosmics debug 2021.05.21
  MonitorElement* SameSectorGEMminusCSCfpThetaPhi; // GEM cosmics debug 2021.05.21
  MonitorElement* gemPosCham32S5NPadPart; // GEM cosmics debug 2021.05.25
  MonitorElement* gemPosCham02S6NPadPart; // GEM cosmics debug 2021.05.25
  MonitorElement* gemNegCham08S1NPadPart; // GEM cosmics debug 2021.05.25
  MonitorElement* gemNegCham20S3NPadPart; // GEM cosmics debug 2021.05.25
  MonitorElement* gemNegCham12PadPart;    // GEM cosmics debug 2021.05.25
  MonitorElement* gemNegBXAddress0134;

  MonitorElement* gemHitBX;
  MonitorElement* gemHitOccupancy;
  MonitorElement* gemHitTiming[5];
  MonitorElement* gemHitTimingFrac[5];
  MonitorElement* gemHitTimingTot;
  MonitorElement* gemChamberPad[2];
  MonitorElement* gemChamberPartition[2];
  MonitorElement* gemChamberAddress[2];
  MonitorElement* gemChamberVFAT[2];
  MonitorElement* gemBXVFAT[2];
  MonitorElement* gemBXVFATC91011[2];
  MonitorElement* gemBXVFATPerChamber[36][2][2];
  MonitorElement* gemBXVFATPerChamberCoincidence[36][2][2];
  MonitorElement* gemBXVFATC9[2];
  MonitorElement* gemBXVFATC10[2];
  MonitorElement* gemBXVFATC11[2];
  MonitorElement* gemChamberVFATBX[2];
  MonitorElement* emtfTrackBXVsGEMHit[3];
  MonitorElement* emtfTrackModeVsGEMBXDiff[2]; // Add mode vs BXdiff comparison 2020.12.07

  // GEM vs CSC 2020.12.06 
  MonitorElement* gemHitPhi[2];
  MonitorElement* gemHitTheta[2];
  MonitorElement* gemHitVScscLCTPhi[2];
  MonitorElement* gemHitVScscLCTTheta[2];
  MonitorElement* gemHitVScscLCTBX[2];
};

#endif
