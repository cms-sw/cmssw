#ifndef TkAlCaRecoMonitor_H
#define TkAlCaRecoMonitor_H
// -*- C++ -*-
//
// Package:    TkAlCaRecoMonitor
// Class:      TkAlCaRecoMonitor
// 
/**\class TkAlCaRecoMonitor TkAlCaRecoMonitor.cc DQM/TrackerMonitorTrack/src/TkAlCaRecoMonitor.cc
Monitoring special quantities related to Tracker Alignment AlCaReco Production.
*/

#include <memory>
#include <fstream>
#include <map>
#include <algorithm>
#include <vector>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

//DataFormats
#include <DataFormats/JetReco/interface/CaloJet.h>


class TrackerGeometry;
class DQMStore;

class TkAlCaRecoMonitor : public DQMEDAnalyzer {
 public:
  explicit TkAlCaRecoMonitor(const edm::ParameterSet&);
  ~TkAlCaRecoMonitor();

  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  
 private:
  void fillHitmaps(const reco::Track &track, const TrackerGeometry& geometry);
  void fillRawIdMap(const TrackerGeometry &geometry);

  // ----------member data ---------------------------
  edm::ParameterSet conf_;

  double maxJetPt_;
  
  //1D
  MonitorElement* invariantMass_;
  MonitorElement* sumCharge_;
  MonitorElement* TrackQuality_;
  MonitorElement* jetPt_;
  MonitorElement* minJetDeltaR_;
  MonitorElement* minTrackDeltaR_;
  MonitorElement* AlCaRecoTrackEfficiency_;
  MonitorElement* Hits_perDetId_;
  MonitorElement* TrackPtPositive_;
  MonitorElement* TrackPtNegative_;
  MonitorElement* TrackCurvature_;
  //2D
  MonitorElement* Hits_ZvsR_;
  MonitorElement* Hits_XvsY_;

  bool fillInvariantMass_;
  bool fillRawIdMap_;
  bool runsOnReco_;
  bool useSignedR_;

  edm::EDGetTokenT<reco::TrackCollection> trackProducer_;
  edm::EDGetTokenT<reco::TrackCollection> referenceTrackProducer_;
  edm::EDGetTokenT<reco::CaloJetCollection> jetCollection_;
  double daughterMass_;
  std::map<int,int> binByRawId_;
};
#endif
