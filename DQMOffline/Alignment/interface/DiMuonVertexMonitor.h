#ifndef DQMOffline_Alignment_DiMuonVertexMonitor_H
#define DQMOffline_Alignment_DiMuonVertexMonitor_H

// -*- C++ -*-
//
// Package:    DiMuonVertexMonitor
// Class:      DiMuonVertexMonitor
//
/**\class DiMuonVertexMonitor DiMuonVertexMonitor.cc
   DQM/TrackerMonitorTrack/src/DiMuonVertexMonitor.cc
   Monitoring  quantities related to the DiMuon vertex during Tracker Alignment AlCaReco Production
*/

// system includes
#include <string>

// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

class DiMuonVertexMonitor : public DQMEDAnalyzer {
public:
  explicit DiMuonVertexMonitor(const edm::ParameterSet &);
  ~DiMuonVertexMonitor() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  const reco::Vertex *findClosestVertex(const TransientVertex aTransVtx, const reco::VertexCollection *vertices) const;

  // ----------member data ---------------------------
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbESToken_;

  //used to select what tracks to read from configuration file
  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  //used to select what vertices to read from configuration file
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  const std::string motherName_;
  const std::string MEFolderName_;  // Top-level folder name
  const bool useClosestVertex_;

  std::pair<float, float> massLimits_; /* for the mass plot x-range */
  const float maxSVdist_;

  // vertex quantities
  MonitorElement *hSVProb_;
  MonitorElement *hSVChi2_;
  MonitorElement *hSVNormChi2_;

  MonitorElement *hSVDist_;
  MonitorElement *hSVDistErr_;
  MonitorElement *hSVDistSig_;
  MonitorElement *hSVCompatibility_;

  MonitorElement *hSVDist3D_;
  MonitorElement *hSVDist3DErr_;
  MonitorElement *hSVDist3DSig_;
  MonitorElement *hSVCompatibility3D_;

  MonitorElement *hCosPhi_;
  MonitorElement *hCosPhi3D_;
  MonitorElement *hCosPhiInv_;
  MonitorElement *hCosPhiInv3D_;
  MonitorElement *hCosPhiUnbalance_;
  MonitorElement *hCosPhi3DUnbalance_;
  MonitorElement *hInvMass_;
  MonitorElement *hCutFlow_;

  // impact parameters information
  MonitorElement *hdxy_;
  MonitorElement *hdz_;
  MonitorElement *hdxyErr_;
  MonitorElement *hdzErr_;
  MonitorElement *hIP2d_;
  MonitorElement *hIP3d_;
  MonitorElement *hIP2dsig_;
  MonitorElement *hIP3dsig_;
};
#endif
