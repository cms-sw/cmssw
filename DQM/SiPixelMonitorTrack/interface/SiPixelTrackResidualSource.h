#ifndef SiPixelTrackResidualSource_H
#define SiPixelTrackResidualSource_H

// Package: SiPixelMonitorTrack
// Class:   SiPixelTrackResidualSource
//
// class SiPixelTrackResidualSource SiPixelTrackResidualSource.h
//       DQM/SiPixelMonitorTrack/interface/SiPixelTrackResidualSource.h
//
// Description:    <one line class summary>
// Implementation: <Notes on implementation>
//
// Original Author: Shan-Huei Chuang
//         Created: Fri Mar 23 18:41:42 CET 2007
//
// Updated by: Lukas Wehrli
// for pixel offline DQM

#include "DQM/SiPixelMonitorTrack/interface/SiPixelTrackResidualModule.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Files added for monitoring track quantities
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include <cstdint>

class SiPixelTrackResidualSource : public DQMEDAnalyzer {
public:
  explicit SiPixelTrackResidualSource(const edm::ParameterSet &);
  ~SiPixelTrackResidualSource() override;

  void dqmBeginRun(const edm::Run &r, edm::EventSetup const &iSetup) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void getrococcupancy(DetId detId,
                       const edm::DetSetVector<PixelDigi> &diginp,
                       const TrackerTopology *const tTopo,
                       std::vector<MonitorElement *> meinput);
  void triplets(double x1,
                double y1,
                double z1,
                double x2,
                double y2,
                double z2,
                double x3,
                double y3,
                double z3,
                double ptsig,
                double &dc,
                double &dz,
                double kap);

  std::string topFolderName_;

private:
  edm::ParameterSet pSet_;
  edm::InputTag src_;
  edm::InputTag clustersrc_;
  edm::InputTag tracksrc_;
  std::string ttrhbuilder_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;
  edm::EDGetTokenT<reco::TrackCollection> generalTracksToken_;
  edm::EDGetTokenT<std::vector<Trajectory>> tracksrcToken_;
  edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> trackAssociationToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> clustersrcToken_;
  std::string vtxsrc_;
  edm::InputTag digisrc_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> digisrcToken_;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackBuilderToken_;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> transientTrackingRecHitBuilderToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoTokenBeginRun_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomTokenBeginRun_;

  bool debug_;
  bool modOn;
  bool reducedSet;
  // barrel:
  bool ladOn, layOn, phiOn;
  // forward:
  bool ringOn, bladeOn, diskOn;
  bool isUpgrade;
  double ptminres_;
  bool firstRun;
  int NTotal;
  int NLowProb;

  std::map<uint32_t, SiPixelTrackResidualModule *> theSiPixelStructure;

  MonitorElement *meSubdetResidualX[3];
  MonitorElement *meSubdetResidualY[3];

  std::vector<MonitorElement *> meResidualXSummedLay;
  std::vector<MonitorElement *> meResidualYSummedLay;

  MonitorElement *meNofTracks_;
  MonitorElement *meNofTracksInPixVol_;
  MonitorElement *meNofClustersOnTrack_;
  std::vector<MonitorElement *> meNofClustersvsPhiOnTrack_layers;
  std::vector<MonitorElement *> meNofClustersvsPhiOnTrack_diskps;
  std::vector<MonitorElement *> meNofClustersvsPhiOnTrack_diskms;
  MonitorElement *meNofClustersNotOnTrack_;
  MonitorElement *meClChargeOnTrack_all;
  MonitorElement *meClChargeOnTrack_bpix;
  MonitorElement *meClChargeOnTrack_fpix;
  std::vector<MonitorElement *> meClChargeOnTrack_layers;
  std::vector<MonitorElement *> meClChargeOnTrack_diskps;
  std::vector<MonitorElement *> meClChargeOnTrack_diskms;
  MonitorElement *meClChargeNotOnTrack_all;
  MonitorElement *meClChargeNotOnTrack_bpix;
  MonitorElement *meClChargeNotOnTrack_fpix;
  std::vector<MonitorElement *> meClChargeNotOnTrack_layers;
  std::vector<MonitorElement *> meClChargeNotOnTrack_diskps;
  std::vector<MonitorElement *> meClChargeNotOnTrack_diskms;
  MonitorElement *meClSizeOnTrack_all;
  MonitorElement *meClSizeOnTrack_bpix;
  MonitorElement *meClSizeOnTrack_fpix;
  std::vector<MonitorElement *> meClSizeOnTrack_layers;
  std::vector<MonitorElement *> meClSizeOnTrack_diskps;
  std::vector<MonitorElement *> meClSizeOnTrack_diskms;
  MonitorElement *meClSizeNotOnTrack_all;
  MonitorElement *meClSizeNotOnTrack_bpix;
  MonitorElement *meClSizeNotOnTrack_fpix;
  std::vector<MonitorElement *> meClSizeNotOnTrack_layers;
  std::vector<MonitorElement *> meClSizeNotOnTrack_diskps;
  std::vector<MonitorElement *> meClSizeNotOnTrack_diskms;
  MonitorElement *meClSizeXOnTrack_all;
  MonitorElement *meClSizeXOnTrack_bpix;
  MonitorElement *meClSizeXOnTrack_fpix;
  std::vector<MonitorElement *> meClSizeXOnTrack_layers;
  std::vector<MonitorElement *> meClSizeXOnTrack_diskps;
  std::vector<MonitorElement *> meClSizeXOnTrack_diskms;
  MonitorElement *meClSizeXNotOnTrack_all;
  MonitorElement *meClSizeXNotOnTrack_bpix;
  MonitorElement *meClSizeXNotOnTrack_fpix;
  std::vector<MonitorElement *> meClSizeXNotOnTrack_layers;
  std::vector<MonitorElement *> meClSizeXNotOnTrack_diskps;
  std::vector<MonitorElement *> meClSizeXNotOnTrack_diskms;
  MonitorElement *meClSizeYOnTrack_all;
  MonitorElement *meClSizeYOnTrack_bpix;
  MonitorElement *meClSizeYOnTrack_fpix;
  std::vector<MonitorElement *> meClSizeYOnTrack_layers;
  std::vector<MonitorElement *> meClSizeYOnTrack_diskps;
  std::vector<MonitorElement *> meClSizeYOnTrack_diskms;
  MonitorElement *meClSizeYNotOnTrack_all;
  MonitorElement *meClSizeYNotOnTrack_bpix;
  MonitorElement *meClSizeYNotOnTrack_fpix;
  std::vector<MonitorElement *> meClSizeYNotOnTrack_layers;
  std::vector<MonitorElement *> meClSizeYNotOnTrack_diskps;
  std::vector<MonitorElement *> meClSizeYNotOnTrack_diskms;

  // new
  MonitorElement *meNClustersOnTrack_all;
  MonitorElement *meNClustersOnTrack_bpix;
  MonitorElement *meNClustersOnTrack_fpix;
  std::vector<MonitorElement *> meNClustersOnTrack_layers;
  std::vector<MonitorElement *> meNClustersOnTrack_diskps;
  std::vector<MonitorElement *> meNClustersOnTrack_diskms;
  MonitorElement *meNClustersNotOnTrack_all;
  MonitorElement *meNClustersNotOnTrack_bpix;
  MonitorElement *meNClustersNotOnTrack_fpix;
  std::vector<MonitorElement *> meNClustersNotOnTrack_layers;
  std::vector<MonitorElement *> meNClustersNotOnTrack_diskps;
  std::vector<MonitorElement *> meNClustersNotOnTrack_diskms;
  //

  std::vector<MonitorElement *> meClPosLayersOnTrack;
  std::vector<MonitorElement *> meClPosLayersLadVsModOnTrack;
  std::vector<MonitorElement *> meClPosLayersNotOnTrack;
  std::vector<MonitorElement *> meClPosDiskspzOnTrack;
  std::vector<MonitorElement *> meClPosDisksmzOnTrack;
  std::vector<MonitorElement *> meClPosDiskspzNotOnTrack;
  std::vector<MonitorElement *> meClPosDisksmzNotOnTrack;

  std::vector<MonitorElement *> meZeroRocLadvsModOnTrackBarrel;
  std::vector<MonitorElement *> meZeroRocLadvsModOffTrackBarrel;

  MonitorElement *meHitProbability;
  MonitorElement *meRocBladevsDiskEndcapOnTrk;
  MonitorElement *meRocBladevsDiskEndcapOffTrk;

  void getepixrococcupancyontrk(const TrackerTopology *const tTopo,
                                TransientTrackingRecHit::ConstRecHitPointer hit,
                                float xclust,
                                float yclust,
                                float z,
                                MonitorElement *meinput);
  void getepixrococcupancyofftrk(
      DetId detId, const TrackerTopology *const tTopo, float xclust, float yclust, float z, MonitorElement *meinput);

  int noOfLayers;
  int noOfDisks;
};

#endif
