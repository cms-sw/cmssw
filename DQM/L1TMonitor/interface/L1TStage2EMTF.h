#ifndef DQM_L1TMonitor_L1TStage2EMTF_h
#define DQM_L1TMonitor_L1TStage2EMTF_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

namespace emtfdqm {
  struct Histograms {
    ConcurrentMonitorElement emtfErrors;
    ConcurrentMonitorElement mpcLinkErrors;
    ConcurrentMonitorElement mpcLinkGood;

    ConcurrentMonitorElement cscLCTBX;
    ConcurrentMonitorElement cscLCTStrip[20];
    ConcurrentMonitorElement cscLCTWire[20];
    ConcurrentMonitorElement cscChamberStrip[20];
    ConcurrentMonitorElement cscChamberWire[20];
    ConcurrentMonitorElement cscLCTOccupancy;
    ConcurrentMonitorElement cscDQMOccupancy;
    ConcurrentMonitorElement cscLCTTiming[5];
    ConcurrentMonitorElement cscTimingTot;

    ConcurrentMonitorElement emtfnTracks;
    ConcurrentMonitorElement emtfTracknHits;
    ConcurrentMonitorElement emtfTrackBX;
    ConcurrentMonitorElement emtfTrackBXVsCSCLCT[3];
    ConcurrentMonitorElement emtfTrackBXVsRPCHit[3];
    ConcurrentMonitorElement emtfTrackPt;
    ConcurrentMonitorElement emtfTrackEta;
    ConcurrentMonitorElement emtfTrackPhi;
    ConcurrentMonitorElement emtfTrackPhiHighQuality;
    ConcurrentMonitorElement emtfTrackOccupancy;
    ConcurrentMonitorElement emtfTrackMode;
    ConcurrentMonitorElement emtfTrackQuality;
    ConcurrentMonitorElement emtfTrackQualityVsMode;

    ConcurrentMonitorElement emtfMuonBX;
    ConcurrentMonitorElement emtfMuonhwPt;
    ConcurrentMonitorElement emtfMuonhwEta;
    ConcurrentMonitorElement emtfMuonhwPhi;
    ConcurrentMonitorElement emtfMuonhwQual;

    ConcurrentMonitorElement rpcHitBX;
    ConcurrentMonitorElement rpcHitOccupancy;
    ConcurrentMonitorElement rpcHitTiming[5];
    ConcurrentMonitorElement rpcHitTimingTot;
    ConcurrentMonitorElement rpcHitPhi[12];
    ConcurrentMonitorElement rpcHitTheta[12];
    ConcurrentMonitorElement rpcChamberPhi[12];
    ConcurrentMonitorElement rpcChamberTheta[12];

    ConcurrentMonitorElement rpcHitTimingInTrack;
  };
}

class L1TStage2EMTF : public DQMGlobalEDAnalyzer<emtfdqm::Histograms> {

 public:

  L1TStage2EMTF(const edm::ParameterSet& ps);
  ~L1TStage2EMTF() override;

 protected:

  void dqmBeginRun(const edm::Run&, const edm::EventSetup&, emtfdqm::Histograms &) const override;
  void bookHistograms(DQMStore::ConcurrentBooker&, const edm::Run&, const edm::EventSetup&, emtfdqm::Histograms &) const override;
  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, emtfdqm::Histograms const&) const override;

 private:

  edm::EDGetTokenT<l1t::EMTFDaqOutCollection> daqToken;
  edm::EDGetTokenT<l1t::EMTFHitCollection> hitToken;
  edm::EDGetTokenT<l1t::EMTFTrackCollection> trackToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> muonToken;
  std::string monitorDir;
  bool verbose;

  const int emtfnTracksNbins;
};

#endif
