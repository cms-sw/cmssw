#ifndef DQMOffline_Alignment_TkAlCaRecoMonitor_H
#define DQMOffline_Alignment_TkAlCaRecoMonitor_H

// -*- C++ -*-
//
// Package:    TkAlCaRecoMonitor
// Class:      TkAlCaRecoMonitor
//
/**\class TkAlCaRecoMonitor TkAlCaRecoMonitor.cc
   DQM/TrackerMonitorTrack/src/TkAlCaRecoMonitor.cc
   Monitoring special quantities related to Tracker Alignment AlCaReco Production.
*/

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class TkAlCaRecoMonitor : public DQMEDAnalyzer {
public:
  explicit TkAlCaRecoMonitor(const edm::ParameterSet &);
  ~TkAlCaRecoMonitor() override = default;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  void fillHitmaps(const reco::Track &track, const TrackerGeometry &geometry);
  void fillRawIdMap(const TrackerGeometry &geometry);

  // ----------member data ---------------------------
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;

  edm::ParameterSet conf_;

  double maxJetPt_;

  // 1D
  MonitorElement *invariantMass_;
  MonitorElement *sumCharge_;
  MonitorElement *TrackQuality_;
  MonitorElement *jetPt_;
  MonitorElement *minJetDeltaR_;
  MonitorElement *minTrackDeltaR_;
  MonitorElement *AlCaRecoTrackEfficiency_;
  MonitorElement *Hits_perDetId_;
  MonitorElement *TrackPtPositive_;
  MonitorElement *TrackPtNegative_;
  MonitorElement *TrackCurvature_;
  // 2D
  MonitorElement *Hits_ZvsR_;
  MonitorElement *Hits_XvsY_;

  bool fillInvariantMass_;
  bool fillRawIdMap_;
  bool runsOnReco_;
  bool useSignedR_;

  edm::EDGetTokenT<reco::TrackCollection> trackProducer_;
  edm::EDGetTokenT<reco::TrackCollection> referenceTrackProducer_;
  edm::EDGetTokenT<reco::CaloJetCollection> jetCollection_;
  double daughterMass_;
  std::map<int, int> binByRawId_;
};
#endif
