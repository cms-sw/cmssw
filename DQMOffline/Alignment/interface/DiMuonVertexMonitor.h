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
  // ----------member data ---------------------------
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbESToken_;
  const edm::EDGetTokenT<reco::TrackCollection>
      tracksToken_;  //used to select what tracks to read from configuration file
  const edm::EDGetTokenT<reco::VertexCollection>
      vertexToken_;                 //used to select what vertices to read from configuration file
  const std::string MEFolderName_;  // Top-level folder name
  const float maxSVdist_;

  // 1D
  MonitorElement *hSVProb_;
  MonitorElement *hSVDist_;
  MonitorElement *hSVDistErr_;
  MonitorElement *hSVDistSig_;
  MonitorElement *hSVDist3D_;
  MonitorElement *hSVDist3DErr_;
  MonitorElement *hSVDist3DSig_;
  MonitorElement *hCosPhi_;
  MonitorElement *hCosPhi3D_;
  MonitorElement *hCosPhiInv_;
  MonitorElement *hCosPhiInv3D_;
  MonitorElement *hTrackInvMass_;
  MonitorElement *hCutFlow_;
};
#endif
