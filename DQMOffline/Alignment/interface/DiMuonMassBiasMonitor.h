#ifndef DQMOffline_Alignment_DiMuonMassBiasMonitor_H
#define DQMOffline_Alignment_DiMuonMassBiasMonitor_H

// -*- C++ -*-
//
// Package:    DiMuonMassBiasMonitor
// Class:      DiMuonMassBiasMonitor
//
/**\class DiMuonMassBiasMonitor DiMuonMassBiasMonitor.cc
   DQM/TrackerMonitorTrack/src/DiMuonMassBiasMonitor.cc
   Monitoring DiMuon mass bias 
*/

// system includes
#include <string>

// user includes
#include "DQMOffline/Alignment/interface/DiLeptonPlotHelpers.h"
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

struct ComponentHists {
  dqm::reco::MonitorElement* h_pt;
  dqm::reco::MonitorElement* h_eta;
  dqm::reco::MonitorElement* h_phi;

  dqm::reco::MonitorElement* h_dxy;
  dqm::reco::MonitorElement* h_exy;
  dqm::reco::MonitorElement* h_dz;
  dqm::reco::MonitorElement* h_ez;

  dqm::reco::MonitorElement* h_chi2;
};

struct DecayHists {
  // kinematics
  dqm::reco::MonitorElement* h_mass;
  dqm::reco::MonitorElement* h_pt;
  dqm::reco::MonitorElement* h_eta;
  dqm::reco::MonitorElement* h_phi;

  // position
  dqm::reco::MonitorElement* h_displ2D;
  dqm::reco::MonitorElement* h_displ3D;
  dqm::reco::MonitorElement* h_sign2D;
  dqm::reco::MonitorElement* h_sign3D;

  // ct and pointing angle
  dqm::reco::MonitorElement* h_ct;
  dqm::reco::MonitorElement* h_pointing;

  // quality
  dqm::reco::MonitorElement* h_vertNormChi2;
  dqm::reco::MonitorElement* h_vertProb;

  std::vector<ComponentHists> decayComponents;
};

class DiMuonMassBiasMonitor : public DQMEDAnalyzer {
public:
  explicit DiMuonMassBiasMonitor(const edm::ParameterSet&);
  ~DiMuonMassBiasMonitor() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void bookDecayHists(DQMStore::IBooker&,
                      DecayHists&,
                      std::string const&,
                      std::string const&,
                      int,
                      float,
                      float,
                      float distanceScaleFactor = 1.) const;

  void bookDecayComponentHistograms(DQMStore::IBooker& ibook, DecayHists& histos) const;

  void bookComponentHists(DQMStore::IBooker&,
                          DecayHists&,
                          TString const&,  // TString for the IBooker interface
                          float distanceScaleFactor = 1.) const;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  reco::Vertex const* fillDecayHistograms(DecayHists const&,
                                          std::vector<const reco::Track*> const& tracks,
                                          const reco::VertexCollection* const& pvs,
                                          const edm::EventSetup&) const;

  void fillComponentHistograms(ComponentHists const& histos,
                               const reco::Track* const& component,
                               reco::BeamSpot const* bs,
                               reco::Vertex const* pv) const;

private:
  // ----------member data ---------------------------
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbESToken_;
  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

  const std::string MEFolderName_;  // Top-level folder name
  const std::string decayMotherName_;
  const double distanceScaleFactor_;
  edm::ParameterSet DiMuMassConfiguration_;

  // 2D histograms of bias vs variable
  DiLepPlotHelp::PlotsVsKinematics ZMassPlots = DiLepPlotHelp::PlotsVsKinematics(DiLepPlotHelp::MM);
  // Decay histograms
  DecayHists histosZmm;
};
#endif
