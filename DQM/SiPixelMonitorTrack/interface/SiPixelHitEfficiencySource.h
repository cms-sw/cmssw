#ifndef SiPixelHitEfficiencySource_H
#define SiPixelHitEfficiencySource_H

// Package: SiPixelMonitorTrack
// Class:   SiPixelHitEfficiencySource
//
// class SiPixelHitEfficiencySource SiPixelHitEfficiencySource.h
//       DQM/SiPixelMonitorTrack/interface/SiPixelHitEfficiencySource.h
//
// Description:    <one line class summary>
// Implementation: <Notes on implementation>
//
//
// Original Authors: Romain Rougny & Luca Mucibello
//         Created: Mar Nov 10 13:29:00 CET 2009

#include "DQM/SiPixelMonitorTrack/interface/SiPixelHitEfficiencyModule.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// Files added for monitoring track quantities
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include <cstdint>

class SiPixelHitEfficiencySource : public DQMEDAnalyzer {
public:
  explicit SiPixelHitEfficiencySource(const edm::ParameterSet &);
  ~SiPixelHitEfficiencySource() override;

  void dqmBeginRun(const edm::Run &r, edm::EventSetup const &iSetup) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void fillClusterProbability(int, int, bool, double);

private:
  edm::ParameterSet pSet_;
  edm::InputTag src_;
  // edm::InputTag tracksrc_;
  edm::EDGetTokenT<reco::VertexCollection> vertexCollectionToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> tracksrc_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> clusterCollectionToken_;

  edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerEventToken_;

  bool applyEdgeCut_;
  double nSigma_EdgeCut_;

  bool debug_;
  bool modOn;
  // barrel:
  bool ladOn, layOn, phiOn;
  // forward:
  bool ringOn, bladeOn, diskOn;

  bool firstRun;

  std::map<uint32_t, SiPixelHitEfficiencyModule *> theSiPixelStructure;

  std::string vtxsrc_;
  int nmissing, nvalid;

  int nvtx_;
  int vtxntrk_;
  double vtxD0_;
  double vtxX_;
  double vtxY_;
  double vtxZ_;
  double vtxndof_;
  double vtxchi2_;

  bool isUpgrade;

  // MEs for cluster probability
  MonitorElement *meClusterProbabilityL1_Plus_;
  MonitorElement *meClusterProbabilityL1_Minus_;

  MonitorElement *meClusterProbabilityL2_Plus_;
  MonitorElement *meClusterProbabilityL2_Minus_;

  MonitorElement *meClusterProbabilityL3_Plus_;
  MonitorElement *meClusterProbabilityL3_Minus_;

  MonitorElement *meClusterProbabilityD1_Plus_;
  MonitorElement *meClusterProbabilityD1_Minus_;

  MonitorElement *meClusterProbabilityD2_Plus_;
  MonitorElement *meClusterProbabilityD2_Minus_;
};

#endif
