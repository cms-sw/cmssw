#ifndef DQM_L1TMonitor_L1TdeStage2EMTF_h
#define DQM_L1TMonitor_L1TdeStage2EMTF_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"


class L1TdeStage2EMTF : public DQMEDAnalyzer {

 public:

  L1TdeStage2EMTF(const edm::ParameterSet& ps);
  virtual ~L1TdeStage2EMTF();

 protected:

  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

 private:

  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> dataToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> emulToken;
  edm::EDGetTokenT<l1t::EMTFHitCollection> datahitToken;
  edm::EDGetTokenT<l1t::EMTFHitCollection> emulhitToken;
  edm::EDGetTokenT<l1t::EMTFTrackCollection> datatrackToken;
  edm::EDGetTokenT<l1t::EMTFTrackCollection> emultrackToken;

  std::string monitorDir;
  bool verbose;

  //Output Track Elements
  MonitorElement* emtfComparenMuonsEvent;
  MonitorElement* emtfMuonMatchhwEta;
  MonitorElement* emtfMuonMatchhwPhi;
  MonitorElement* emtfMuonMatchhwPt;
  MonitorElement* emtfMuonMatchhwQual;



  //Track Elements
  MonitorElement* emtfTrackMatchEta;
  MonitorElement* emtfTrackMatchPhi;
  MonitorElement* emtfTrackMatchPt;
  MonitorElement* emtfTrackMatchBx;
  MonitorElement* emtfTrackMatchQuality;
  MonitorElement* emtfTrackMatchMode;
 
  MonitorElement* emtfTrackEtaDif;
  MonitorElement* emtfTrackPhiDif;
  MonitorElement* emtfTrackPtDif;
  MonitorElement* emtfTrackQualDif;

};

#endif
