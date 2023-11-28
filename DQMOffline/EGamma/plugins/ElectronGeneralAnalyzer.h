#ifndef DQMOffline_EGamma_ElectronGeneralAnalyzer_h
#define DQMOffline_EGamma_ElectronGeneralAnalyzer_h

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

class MagneticField;

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

class ElectronGeneralAnalyzer : public ElectronDqmAnalyzerBase {
public:
  explicit ElectronGeneralAnalyzer(const edm::ParameterSet &conf);
  ~ElectronGeneralAnalyzer() override;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

private:
  //=========================================
  // parameters
  //=========================================

  // collection input tags
  edm::EDGetTokenT<reco::GsfElectronCollection> electronCollection_;
  edm::EDGetTokenT<reco::SuperClusterCollection> matchingObjectCollection_;
  edm::EDGetTokenT<reco::GsfTrackCollection> gsftrackCollection_;
  edm::EDGetTokenT<reco::TrackCollection> trackCollection_;
  edm::EDGetTokenT<reco::VertexCollection> vertexCollection_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;

  //=========================================
  // histograms
  //=========================================

  MonitorElement *h2_ele_beamSpotXvsY;
  MonitorElement *py_ele_nElectronsVsLs;
  MonitorElement *py_ele_nClustersVsLs;
  MonitorElement *py_ele_nGsfTracksVsLs;
  MonitorElement *py_ele_nTracksVsLs;
  MonitorElement *py_ele_nVerticesVsLs;
};

#endif
