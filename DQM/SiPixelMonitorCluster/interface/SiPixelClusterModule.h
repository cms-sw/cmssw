#ifndef SiPixelMonitorCluster_SiPixelClusterModule_h
#define SiPixelMonitorCluster_SiPixelClusterModule_h
// -*- C++ -*-
//
// Package:    SiPixelMonitorDigi
// Class:      SiPixelClusterModule
// 
/*

 Description: Cluster monitoring elements for a Pixel sensor

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia & Andrew York
//         Created:  
//
//
//  Updated by: Lukas Wehrli
//  for pixel offline DQM 
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <boost/cstdint.hpp>

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h" 
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h" 
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

class SiPixelClusterModule {        

 public:

  /// Default constructor
  SiPixelClusterModule();
  /// Constructor with raw DetId
  SiPixelClusterModule(const uint32_t& id);
  /// Constructor with raw DetId and sensor size
  SiPixelClusterModule(const uint32_t& id, const int& ncols, const int& nrows);
  /// Destructor
  ~SiPixelClusterModule();

  typedef edmNew::DetSet<SiPixelCluster>::const_iterator    ClusterIterator;

  /// Book histograms
  void book(const edm::ParameterSet& iConfig, const edm::EventSetup& iSetup, DQMStore::IBooker & iBooker, int type=0, bool twoD=true, bool reducedSet=false, bool isUpgrade=false);
  /// Fill histograms
  int fill(const edmNew::DetSetVector<SiPixelCluster> & input, 
            const TrackerGeometry* tracker,
	    std::vector<MonitorElement*>& layers,
	    std::vector<MonitorElement*>& diskspz,
	    std::vector<MonitorElement*>& disksmz,
            bool modon=true, 
	    bool ladon=false, 
	    bool layon=false, 
	    bool phion=false, 
	    bool bladeon=false, 
	    bool diskon=false, 
	    bool ringon=false, 
	    bool twoD=true,
	    bool reducedSet=false,
	    bool smileyon=false,
	    bool isUpgrade=false);
  
 private:

  const TrackerTopology *pTT;
  uint32_t id_;
  int ncols_;
  int nrows_;
  MonitorElement* meNClusters_;
  MonitorElement* meY_;
  MonitorElement* meX_;
  MonitorElement* meCharge_;
  MonitorElement* meSize_;
  MonitorElement* meSizeX_;
  MonitorElement* meSizeY_;
  MonitorElement* meMinRow_;
  MonitorElement* meMaxRow_;
  MonitorElement* meMinCol_;
  MonitorElement* meMaxCol_;
  MonitorElement* mePixClusters_;
  MonitorElement* mePixClusters_px_;
  MonitorElement* mePixClusters_py_;
  //  MonitorElement* meEdgeHitX_;
  //  MonitorElement* meEdgeHitY_;
  MonitorElement* meClPosLayer1;
  MonitorElement* meClPosLayer2;
  MonitorElement* meClPosLayer3;
  MonitorElement* meClPosLayer4;
  MonitorElement* meClPosDisk1pz;
  MonitorElement* meClPosDisk2pz;
  MonitorElement* meClPosDisk3pz;
  MonitorElement* meClPosDisk1mz;
  MonitorElement* meClPosDisk2mz;
  MonitorElement* meClPosDisk3mz;
  
  //barrel
  MonitorElement* meNClustersLad_;
  MonitorElement* meYLad_;
  MonitorElement* meXLad_;
  MonitorElement* meChargeLad_;
  MonitorElement* meSizeLad_;
  MonitorElement* meSizeXLad_;
  MonitorElement* meSizeYLad_;
  MonitorElement* meMinRowLad_;
  MonitorElement* meMaxRowLad_;
  MonitorElement* meMinColLad_;
  MonitorElement* meMaxColLad_;
  MonitorElement* mePixClustersLad_;
  MonitorElement* mePixClustersLad_px_;
  MonitorElement* mePixClustersLad_py_;

  MonitorElement* meSizeYvsEtaBarrel_; 

  MonitorElement* meNClustersLay_;
  MonitorElement* meYLay_;
  MonitorElement* meXLay_;
  MonitorElement* meChargeLay_;
  MonitorElement* meSizeLay_;
  MonitorElement* meSizeXLay_;
  MonitorElement* meSizeYLay_;
  MonitorElement* meMinRowLay_;
  MonitorElement* meMaxRowLay_;
  MonitorElement* meMinColLay_;
  MonitorElement* meMaxColLay_;
  MonitorElement* mePixClustersLay_;
  MonitorElement* mePixClustersLay_px_;
  MonitorElement* mePixClustersLay_py_;

  MonitorElement* meNClustersPhi_;
  MonitorElement* meYPhi_;
  MonitorElement* meXPhi_;
  MonitorElement* meChargePhi_;
  MonitorElement* meSizePhi_;
  MonitorElement* meSizeXPhi_;
  MonitorElement* meSizeYPhi_;
  MonitorElement* meMinRowPhi_;
  MonitorElement* meMaxRowPhi_;
  MonitorElement* meMinColPhi_;
  MonitorElement* meMaxColPhi_;
  MonitorElement* mePixClustersPhi_;
  MonitorElement* mePixClustersPhi_px_;
  MonitorElement* mePixClustersPhi_py_;

  //forward
  MonitorElement* meNClustersBlade_;
  MonitorElement* meYBlade_;
  MonitorElement* meXBlade_;
  MonitorElement* meChargeBlade_;
  MonitorElement* meSizeBlade_;
  MonitorElement* meSizeXBlade_;
  MonitorElement* meSizeYBlade_;
  MonitorElement* meMinRowBlade_;
  MonitorElement* meMaxRowBlade_;
  MonitorElement* meMinColBlade_;
  MonitorElement* meMaxColBlade_;


  MonitorElement* meNClustersDisk_;
  MonitorElement* meYDisk_;
  MonitorElement* meXDisk_;
  MonitorElement* meChargeDisk_;
  MonitorElement* meSizeDisk_;
  MonitorElement* meSizeXDisk_;
  MonitorElement* meSizeYDisk_;
  MonitorElement* meMinRowDisk_;
  MonitorElement* meMaxRowDisk_;
  MonitorElement* meMinColDisk_;
  MonitorElement* meMaxColDisk_;


  MonitorElement* meNClustersRing_;
  MonitorElement* meYRing_;
  MonitorElement* meXRing_;
  MonitorElement* meChargeRing_;
  MonitorElement* meSizeRing_;
  MonitorElement* meSizeXRing_;
  MonitorElement* meSizeYRing_;
  MonitorElement* meMinRowRing_;
  MonitorElement* meMaxRowRing_;
  MonitorElement* meMinColRing_;
  MonitorElement* meMaxColRing_;
  MonitorElement* mePixClustersRing_;
  MonitorElement* mePixClustersRing_px_;
  MonitorElement* mePixClustersRing_py_;

};
#endif
