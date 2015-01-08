// -*- C++ -*-
//
// Package:    SiPixelMonitorCluster
// Class:      SiPixelClusterSource
// 
/**\class 

 Description: Pixel DQM source for Clusters

 Implementation:
     Note that the x- and y-directions referred to in the cluster description refer to local x- and y-values given by the clusterizer.  Local x corresponds to row value and local y corresponds to column value.
*/
//
// Original Author:  Vincenzo Chiochia & Andrew York
//         Created:  
//
//
// Updated by: Lukas Wehrli
// for pixel offline DQM 
#include "DQM/SiPixelMonitorCluster/interface/SiPixelClusterModule.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
/// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <stdlib.h>

// Data Formats
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
//
// Constructors
//
SiPixelClusterModule::SiPixelClusterModule() : id_(0),
					 ncols_(416),
					 nrows_(160) { }
///
SiPixelClusterModule::SiPixelClusterModule(const uint32_t& id) : 
  id_(id),
  ncols_(416),
  nrows_(160)
{ 
}
///
SiPixelClusterModule::SiPixelClusterModule(const uint32_t& id, const int& ncols, const int& nrows) : 
  id_(id),
  ncols_(ncols),
  nrows_(nrows)
{ 
}
//
// Destructor
//
SiPixelClusterModule::~SiPixelClusterModule() {}
//
// Book histograms
//
void SiPixelClusterModule::book(const edm::ParameterSet& iConfig, const edm::EventSetup& iSetup, DQMStore::IBooker & iBooker, int type, bool twoD, bool reducedSet, bool isUpgrade) {
  
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology *pTT = tTopoHandle.product();

  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  bool isHalfModule = false;
  if(barrel){
    isHalfModule = PixelBarrelName(DetId(id_),pTT,isUpgrade).isHalfModule();
  }
  int nbinx = ncols_/2;
  int nbiny = nrows_/2;

  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = iConfig.getParameter<edm::InputTag>( "src" );
  if(type==0){
    SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );
    if(!reducedSet){
      // Lowest cluster row
      hid = theHistogramId->setHistoId("minrow",id_);
      meMinRow_ = iBooker.book1D(hid,"Lowest cluster row",200,0.,200.);
      meMinRow_->setAxisTitle("Lowest cluster row",1);
      // Highest cluster row
      hid = theHistogramId->setHistoId("maxrow",id_);
      meMaxRow_ = iBooker.book1D(hid,"Highest cluster row",200,0.,200.);
      meMaxRow_->setAxisTitle("Highest cluster row",1);
      // Lowest cluster column
      hid = theHistogramId->setHistoId("mincol",id_);
      meMinCol_ = iBooker.book1D(hid,"Lowest cluster column",500,0.,500.);
      meMinCol_->setAxisTitle("Lowest cluster column",1);
      // Highest cluster column
      hid = theHistogramId->setHistoId("maxcol",id_);
      meMaxCol_ = iBooker.book1D(hid,"Highest cluster column",500,0.,500.);
      meMaxCol_->setAxisTitle("Highest cluster column",1);
      // Cluster barycenter X position
      hid = theHistogramId->setHistoId("x",id_);
      meX_ = iBooker.book1D(hid,"Cluster barycenter X (row #)",200,0.,200.);
      meX_->setAxisTitle("Barycenter x-position [row #]",1);
      // Cluster barycenter Y position
      hid = theHistogramId->setHistoId("y",id_);
      meY_ = iBooker.book1D(hid,"Cluster barycenter Y (column #)",500,0.,500.);
      meY_->setAxisTitle("Barycenter y-position [column #]",1);
      // Cluster width on the x-axis
      hid = theHistogramId->setHistoId("sizeX",id_);
      meSizeX_ = iBooker.book1D(hid,"Cluster x-width (rows)",10,0.,10.);
      meSizeX_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      hid = theHistogramId->setHistoId("sizeY",id_);
      meSizeY_ = iBooker.book1D(hid,"Cluster y-width (columns)",15,0.,15.);
      meSizeY_->setAxisTitle("Cluster y-size [columns]",1);
      int nbinx = ncols_/2;
      int nbiny = nrows_/2;
      hid = theHistogramId->setHistoId("hitmap",id_);
      if(twoD){
        // 2D hit map
        mePixClusters_ = iBooker.book2D(hid,"Number of Clusters (1bin=four pixels)",nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
        mePixClusters_->setAxisTitle("Columns",1);
        mePixClusters_->setAxisTitle("Rows",2);
      }else{
        // projections of hitmap
        mePixClusters_px_ = iBooker.book1D(hid+"_px","Number of Clusters (1bin=two columns)",nbinx,0.,float(ncols_));
        mePixClusters_py_ = iBooker.book1D(hid+"_py","Number of Clusters (1bin=two rows)",nbiny,0.,float(nrows_));
        mePixClusters_px_->setAxisTitle("Columns",1);
        mePixClusters_py_->setAxisTitle("Rows",1);
      }
    }
    delete theHistogramId;
  }

  //**
  if(barrel && type==7){
    hid = src.label() + "_Barrel";
    meSizeYvsEtaBarrel_= iBooker.book2D("sizeYvsEta_" + hid,"Cluster size along beamline vs. Cluster position #eta",60,-3.,3.,40,0.,40.);
    meSizeYvsEtaBarrel_->setAxisTitle("Cluster #eta",1);
    meSizeYvsEtaBarrel_->setAxisTitle("Cluster size along beamline [number of pixels]",2);
  }
  if(type==1 && barrel){
    uint32_t DBladder;
    DBladder = PixelBarrelName(DetId(id_),pTT,isUpgrade).ladderName();
    char sladder[80]; sprintf(sladder,"Ladder_%02i",DBladder);
    hid = src.label() + "_" + sladder;
    if(isHalfModule) hid += "H";
    else hid += "F";
    // Number of clusters
    meNClustersLad_ = iBooker.book1D("nclusters_" + hid,"Number of Clusters",8,0.,8.);
    meNClustersLad_->setAxisTitle("Number of Clusters",1);
    // Total cluster charge in MeV
    meChargeLad_ = iBooker.book1D("charge_" + hid,"Cluster charge",100,0.,200.);
    meChargeLad_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSizeLad_ = iBooker.book1D("size_" + hid,"Total cluster size",30,0.,30.);
    meSizeLad_->setAxisTitle("Cluster size [number of pixels]",1);
    if(!reducedSet){
      // Lowest cluster row
      meMinRowLad_ = iBooker.book1D("minrow_" + hid,"Lowest cluster row",200,0.,200.);
      meMinRowLad_->setAxisTitle("Lowest cluster row",1);
      // Highest cluster row
      meMaxRowLad_ = iBooker.book1D("maxrow_" + hid,"Highest cluster row",200,0.,200.);
      meMaxRowLad_->setAxisTitle("Highest cluster row",1);
      // Lowest cluster column
      meMinColLad_ = iBooker.book1D("mincol_" + hid,"Lowest cluster column",500,0.,500.);
      meMinColLad_->setAxisTitle("Lowest cluster column",1);
      // Highest cluster column
      meMaxColLad_ = iBooker.book1D("maxcol_" + hid,"Highest cluster column",500,0.,500.);
      meMaxColLad_->setAxisTitle("Highest cluster column",1);
      // Cluster barycenter X position
      meXLad_ = iBooker.book1D("x_" + hid,"Cluster barycenter X (row #)",200,0.,200.);
      meXLad_->setAxisTitle("Barycenter x-position [row #]",1);
      // Cluster barycenter Y position
      meYLad_ = iBooker.book1D("y_" + hid,"Cluster barycenter Y (column #)",500,0.,500.);
      meYLad_->setAxisTitle("Barycenter y-position [column #]",1);
      // Cluster width on the x-axis
      meSizeXLad_ = iBooker.book1D("sizeX_" + hid,"Cluster x-width (rows)",10,0.,10.);
      meSizeXLad_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeYLad_ = iBooker.book1D("sizeY_" + hid,"Cluster y-width (columns)",15,0.,15.);
      meSizeYLad_->setAxisTitle("Cluster y-size [columns]",1);
      if(twoD){
        // 2D hit map
        mePixClustersLad_ = iBooker.book2D("hitmap_" + hid,"Number of Clusters (1bin=four pixels)",nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
        mePixClustersLad_->setAxisTitle("Columns",1);
        mePixClustersLad_->setAxisTitle("Rows",2);
      }else{
        // projections of hitmap
        mePixClustersLad_px_ = iBooker.book1D("hitmap_" + hid+"_px","Number of Clusters (1bin=two columns)",nbinx,0.,float(ncols_));
        mePixClustersLad_py_ = iBooker.book1D("hitmap_" + hid+"_py","Number of Clusters (1bin=two rows)",nbiny,0.,float(nrows_));
        mePixClustersLad_px_->setAxisTitle("Columns",1);
        mePixClustersLad_py_->setAxisTitle("Rows",1);
      }
    }
  }

  if(type==2 && barrel){
    
    uint32_t DBlayer;
    DBlayer = PixelBarrelName(DetId(id_),pTT,isUpgrade).layerName();
    char slayer[80]; sprintf(slayer,"Layer_%i",DBlayer);
    hid = src.label() + "_" + slayer;
    // Number of clusters
    meNClustersLay_ = iBooker.book1D("nclusters_" + hid,"Number of Clusters",8,0.,8.);
    meNClustersLay_->setAxisTitle("Number of Clusters",1);
    // Total cluster charge in MeV
    meChargeLay_ = iBooker.book1D("charge_" + hid,"Cluster charge",100,0.,200.);
    meChargeLay_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSizeLay_ = iBooker.book1D("size_" + hid,"Total cluster size",30,0.,30.);
    meSizeLay_->setAxisTitle("Cluster size [in pixels]",1);
    if(!reducedSet){
      // Lowest cluster row
      meMinRowLay_ = iBooker.book1D("minrow_" + hid,"Lowest cluster row",200,0.,200.);
      meMinRowLay_->setAxisTitle("Lowest cluster row",1);
      // Highest cluster row
      meMaxRowLay_ = iBooker.book1D("maxrow_" + hid,"Highest cluster row",200,0.,200.);
      meMaxRowLay_->setAxisTitle("Highest cluster row",1);
      // Lowest cluster column
      meMinColLay_ = iBooker.book1D("mincol_" + hid,"Lowest cluster column",500,0.,500.);
      meMinColLay_->setAxisTitle("Lowest cluster column",1);
      // Highest cluster column
      meMaxColLay_ = iBooker.book1D("maxcol_" + hid,"Highest cluster column",500,0.,500.);
      meMaxColLay_->setAxisTitle("Highest cluster column",1);
      // Cluster barycenter X position
      meXLay_ = iBooker.book1D("x_" + hid,"Cluster barycenter X (row #)",200,0.,200.);
      meXLay_->setAxisTitle("Barycenter x-position [row #]",1);
      // Cluster barycenter Y position
      meYLay_ = iBooker.book1D("y_" + hid,"Cluster barycenter Y (column #)",500,0.,500.);
      meYLay_->setAxisTitle("Barycenter y-position [column #]",1);
      // Cluster width on the x-axis
      meSizeXLay_ = iBooker.book1D("sizeX_" + hid,"Cluster x-width (rows)",10,0.,10.);
      meSizeXLay_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeYLay_ = iBooker.book1D("sizeY_" + hid,"Cluster y-width (columns)",15,0.,15.);
      meSizeYLay_->setAxisTitle("Cluster y-size [columns]",1);
      if(twoD){
        // 2D hit map
        if(isHalfModule){
	  mePixClustersLay_ = iBooker.book2D("hitmap_" + hid,"Number of Clusters (1bin=four pixels)",nbinx,0.,float(ncols_),2*nbiny,0.,float(2*nrows_));
        }else{
	  mePixClustersLay_ = iBooker.book2D("hitmap_" + hid,"Number of Clusters (1bin=four pixels)",nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
        }
        mePixClustersLay_->setAxisTitle("Columns",1);
        mePixClustersLay_->setAxisTitle("Rows",2);
      }else{
        // projections of hitmap
        mePixClustersLay_px_ = iBooker.book1D("hitmap_" + hid+"_px","Number of Clusters (1bin=two columns)",nbinx,0.,float(ncols_));
        if(isHalfModule){
	  mePixClustersLay_py_ = iBooker.book1D("hitmap_" + hid+"_py","Number of Clusters (1bin=two rows)",2*nbiny,0.,float(2*nrows_));
        }else{
	  mePixClustersLay_py_ = iBooker.book1D("hitmap_" + hid+"_py","Number of Clusters (1bin=two rows)",nbiny,0.,float(nrows_));
        }
        mePixClustersLay_px_->setAxisTitle("Columns",1);
        mePixClustersLay_py_->setAxisTitle("Rows",1);
      }
    }
  }
  if(type==3 && barrel){
    uint32_t DBmodule;
    DBmodule = PixelBarrelName(DetId(id_),pTT,isUpgrade).moduleName();
    char smodule[80]; sprintf(smodule,"Ring_%i",DBmodule);
    hid = src.label() + "_" + smodule;
    // Number of clusters
    meNClustersPhi_ = iBooker.book1D("nclusters_" + hid,"Number of Clusters",8,0.,8.);
    meNClustersPhi_->setAxisTitle("Number of Clusters",1);
    // Total cluster charge in MeV
    meChargePhi_ = iBooker.book1D("charge_" + hid,"Cluster charge",100,0.,200.);
    meChargePhi_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSizePhi_ = iBooker.book1D("size_" + hid,"Total cluster size",30,0.,30.);
    meSizePhi_->setAxisTitle("Cluster size [number of pixels]",1);
    if(!reducedSet){
      // Lowest cluster row
      meMinRowPhi_ = iBooker.book1D("minrow_" + hid,"Lowest cluster row",200,0.,200.);
      meMinRowPhi_->setAxisTitle("Lowest cluster row",1);
      // Highest cluster row
      meMaxRowPhi_ = iBooker.book1D("maxrow_" + hid,"Highest cluster row",200,0.,200.);
      meMaxRowPhi_->setAxisTitle("Highest cluster row",1);
      // Lowest cluster column
      meMinColPhi_ = iBooker.book1D("mincol_" + hid,"Lowest cluster column",500,0.,500.);
      meMinColPhi_->setAxisTitle("Lowest cluster column",1);
      // Highest cluster column
      meMaxColPhi_ = iBooker.book1D("maxcol_" + hid,"Highest cluster column",500,0.,500.);
      meMaxColPhi_->setAxisTitle("Highest cluster column",1);
      // Cluster barycenter X position
      meXPhi_ = iBooker.book1D("x_" + hid,"Cluster barycenter X (row #)",200,0.,200.);
      meXPhi_->setAxisTitle("Barycenter x-position [row #]",1);
      // Cluster barycenter Y position
      meYPhi_ = iBooker.book1D("y_" + hid,"Cluster barycenter Y (column #)",500,0.,500.);
      meYPhi_->setAxisTitle("Barycenter y-position [column #]",1);
      // Cluster width on the x-axis
      meSizeXPhi_ = iBooker.book1D("sizeX_" + hid,"Cluster x-width (rows)",10,0.,10.);
      meSizeXPhi_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeYPhi_ = iBooker.book1D("sizeY_" + hid,"Cluster y-width (columns)",15,0.,15.);
      meSizeYPhi_->setAxisTitle("Cluster y-size [columns]",1);
      if(twoD){
        // 2D hit map
        if(isHalfModule){
	  mePixClustersPhi_ = iBooker.book2D("hitmap_" + hid,"Number of Clusters (1bin=four pixels)",nbinx,0.,float(ncols_),2*nbiny,0.,float(2*nrows_));
        }else{
	  mePixClustersPhi_ = iBooker.book2D("hitmap_" + hid,"Number of Clusters (1bin=four pixels)",nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
        }
        mePixClustersPhi_->setAxisTitle("Columns",1);
        mePixClustersPhi_->setAxisTitle("Rows",2);
      }else{
        // projections of hitmap
        mePixClustersPhi_px_ = iBooker.book1D("hitmap_" + hid+"_px","Number of Clusters (1bin=two columns)",nbinx,0.,float(ncols_));
        if(isHalfModule){
	  mePixClustersPhi_py_ = iBooker.book1D("hitmap_" + hid+"_py","Number of Clusters (1bin=two rows)",2*nbiny,0.,float(2*nrows_));
        }else{
	  mePixClustersPhi_py_ = iBooker.book1D("hitmap_" + hid+"_py","Number of Clusters (1bin=two rows)",nbiny,0.,float(nrows_));
        }
        mePixClustersPhi_px_->setAxisTitle("Columns",1);
        mePixClustersPhi_py_->setAxisTitle("Rows",1);
      }
    }
  }

  if(type==4 && endcap){
    uint32_t blade;
    blade = PixelEndcapName(DetId(id_),pTT,isUpgrade).bladeName();
    
    char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
    hid = src.label() + "_" + sblade;
    // Number of clusters
    meNClustersBlade_ = iBooker.book1D("nclusters_" + hid,"Number of Clusters",8,0.,8.);
    meNClustersBlade_->setAxisTitle("Number of Clusters",1);
    // Total cluster charge in MeV
    meChargeBlade_ = iBooker.book1D("charge_" + hid,"Cluster charge",100,0.,200.);
    meChargeBlade_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSizeBlade_ = iBooker.book1D("size_" + hid,"Total cluster size",30,0.,30.);
    meSizeBlade_->setAxisTitle("Cluster size [number of pixels]",1);
    if(!reducedSet){
      // Lowest cluster row
      meMinRowBlade_ = iBooker.book1D("minrow_" + hid,"Lowest cluster row",200,0.,200.);
      meMinRowBlade_->setAxisTitle("Lowest cluster row",1);
      // Highest cluster row
      meMaxRowBlade_ = iBooker.book1D("maxrow_" + hid,"Highest cluster row",200,0.,200.);
      meMaxRowBlade_->setAxisTitle("Highest cluster row",1);
      // Lowest cluster column
      meMinColBlade_ = iBooker.book1D("mincol_" + hid,"Lowest cluster column",500,0.,500.);
      meMinColBlade_->setAxisTitle("Lowest cluster column",1);
      // Highest cluster column
      meMaxColBlade_ = iBooker.book1D("maxcol_" + hid,"Highest cluster column",500,0.,500.);
      meMaxColBlade_->setAxisTitle("Highest cluster column",1);
      // Cluster barycenter X position
      meXBlade_ = iBooker.book1D("x_" + hid,"Cluster barycenter X (row #)",200,0.,200.);
      meXBlade_->setAxisTitle("Barycenter x-position [row #]",1);
      // Cluster barycenter Y position
      meYBlade_ = iBooker.book1D("y_" + hid,"Cluster barycenter Y (column #)",500,0.,500.);
      meYBlade_->setAxisTitle("Barycenter y-position [column #]",1);
      // Cluster width on the x-axis
      meSizeXBlade_ = iBooker.book1D("sizeX_" + hid,"Cluster x-width (rows)",10,0.,10.);
      meSizeXBlade_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeYBlade_ = iBooker.book1D("sizeY_" + hid,"Cluster y-width (columns)",15,0.,15.);
      meSizeYBlade_->setAxisTitle("Cluster y-size [columns]",1);
    }
  }
  if(type==5 && endcap){
    uint32_t disk;
    disk = PixelEndcapName(DetId(id_),pTT,isUpgrade).diskName();
    
    char sdisk[80]; sprintf(sdisk, "Disk_%i",disk);
    hid = src.label() + "_" + sdisk;
    // Number of clusters
    meNClustersDisk_ = iBooker.book1D("nclusters_" + hid,"Number of Clusters",8,0.,8.);
    meNClustersDisk_->setAxisTitle("Number of Clusters",1);
    // Total cluster charge in MeV
    meChargeDisk_ = iBooker.book1D("charge_" + hid,"Cluster charge",100,0.,200.);
    meChargeDisk_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSizeDisk_ = iBooker.book1D("size_" + hid,"Total cluster size",30,0.,30.);
    meSizeDisk_->setAxisTitle("Cluster size [number of pixels]",1);
    if(!reducedSet){
      // Lowest cluster row
      meMinRowDisk_ = iBooker.book1D("minrow_" + hid,"Lowest cluster row",200,0.,200.);
      meMinRowDisk_->setAxisTitle("Lowest cluster row",1);
      // Highest cluster row
      meMaxRowDisk_ = iBooker.book1D("maxrow_" + hid,"Highest cluster row",200,0.,200.);
      meMaxRowDisk_->setAxisTitle("Highest cluster row",1);
      // Lowest cluster column
      meMinColDisk_ = iBooker.book1D("mincol_" + hid,"Lowest cluster column",500,0.,500.);
      meMinColDisk_->setAxisTitle("Lowest cluster column",1);
      // Highest cluster column
      meMaxColDisk_ = iBooker.book1D("maxcol_" + hid,"Highest cluster column",500,0.,500.);
      meMaxColDisk_->setAxisTitle("Highest cluster column",1);
      // Cluster barycenter X position
      meXDisk_ = iBooker.book1D("x_" + hid,"Cluster barycenter X (row #)",200,0.,200.);
      meXDisk_->setAxisTitle("Barycenter x-position [row #]",1);
      // Cluster barycenter Y position
      meYDisk_ = iBooker.book1D("y_" + hid,"Cluster barycenter Y (column #)",500,0.,500.);
      meYDisk_->setAxisTitle("Barycenter y-position [column #]",1);
      // Cluster width on the x-axis
      meSizeXDisk_ = iBooker.book1D("sizeX_" + hid,"Cluster x-width (rows)",10,0.,10.);
      meSizeXDisk_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeYDisk_ = iBooker.book1D("sizeY_" + hid,"Cluster y-width (columns)",15,0.,15.);
      meSizeYDisk_->setAxisTitle("Cluster y-size [columns]",1);
    }
  }

  if(type==6 && endcap){
    uint32_t panel;
    uint32_t module;
    panel= PixelEndcapName(DetId(id_),pTT,isUpgrade).pannelName();
    module= PixelEndcapName(DetId(id_),pTT,isUpgrade).plaquetteName();
    
    char slab[80]; sprintf(slab, "Panel_%i_Ring_%i",panel, module);
    hid = src.label() + "_" + slab;
    // Number of clusters
    meNClustersRing_ = iBooker.book1D("nclusters_" + hid,"Number of Clusters",8,0.,8.);
    meNClustersRing_->setAxisTitle("Number of Clusters",1);
    // Total cluster charge in MeV
    meChargeRing_ = iBooker.book1D("charge_" + hid,"Cluster charge",100,0.,200.);
    meChargeRing_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSizeRing_ = iBooker.book1D("size_" + hid,"Total cluster size",30,0.,30.);
    meSizeRing_->setAxisTitle("Cluster size [number of pixels]",1);
    if(!reducedSet){
      // Lowest cluster row
      meMinRowRing_ = iBooker.book1D("minrow_" + hid,"Lowest cluster row",200,0.,200.);
      meMinRowRing_->setAxisTitle("Lowest cluster row",1);
      // Highest cluster row
      meMaxRowRing_ = iBooker.book1D("maxrow_" + hid,"Highest cluster row",200,0.,200.);
      meMaxRowRing_->setAxisTitle("Highest cluster row",1);
      // Lowest cluster column
      meMinColRing_ = iBooker.book1D("mincol_" + hid,"Lowest cluster column",500,0.,500.);
      meMinColRing_->setAxisTitle("Lowest cluster column",1);
      // Highest cluster column
      meMaxColRing_ = iBooker.book1D("maxcol_" + hid,"Highest cluster column",500,0.,500.);
      meMaxColRing_->setAxisTitle("Highest cluster column",1);
      // Cluster barycenter X position
      meXRing_ = iBooker.book1D("x_" + hid,"Cluster barycenter X (row #)",200,0.,200.);
      meXRing_->setAxisTitle("Barycenter x-position [row #]",1);
      // Cluster barycenter Y position
      meYRing_ = iBooker.book1D("y_" + hid,"Cluster barycenter Y (column #)",500,0.,500.);
      meYRing_->setAxisTitle("Barycenter y-position [column #]",1);
      // Cluster width on the x-axis
      meSizeXRing_ = iBooker.book1D("sizeX_" + hid,"Cluster x-width (rows)",10,0.,10.);
      meSizeXRing_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeYRing_ = iBooker.book1D("sizeY_" + hid,"Cluster y-width (columns)",15,0.,15.);
      meSizeYRing_->setAxisTitle("Cluster y-size [columns]",1);
      if(twoD){
        // 2D hit map
        mePixClustersRing_ = iBooker.book2D("hitmap_" + hid,"Number of Clusters (1bin=four pixels)",nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
        mePixClustersRing_->setAxisTitle("Columns",1);
        mePixClustersRing_->setAxisTitle("Rows",2);
      }else{
        // projections of hitmap
        mePixClustersRing_px_ = iBooker.book1D("hitmap_" + hid+"_px","Number of Clusters (1bin=two columns)",nbinx,0.,float(ncols_));
        mePixClustersRing_py_ = iBooker.book1D("hitmap_" + hid+"_py","Number of Clusters (1bin=two rows)",nbiny,0.,float(nrows_));
        mePixClustersRing_px_->setAxisTitle("Columns",1);
        mePixClustersRing_py_->setAxisTitle("Rows",1);
      }
    }
  }

}
//
// Fill histograms
//
int SiPixelClusterModule::fill(const edmNew::DetSetVector<SiPixelCluster>& input, const TrackerGeometry* tracker,std::vector<MonitorElement*>& layers,std::vector<MonitorElement*>& diskspz,std::vector<MonitorElement*>& disksmz,bool modon, bool ladon, bool layon, bool phion, bool bladeon, bool diskon, bool ringon, bool twoD, bool reducedSet, bool smileyon, bool isUpgrade) {
  
  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  
  edmNew::DetSetVector<SiPixelCluster>::const_iterator isearch = input.find(id_); // search  clusters of detid
  unsigned int numberOfClusters = 0;
  unsigned int numberOfFpixClusters = 0;
  
  if( isearch != input.end() ) {  // Not an empty iterator

    
    // Look at clusters now
    edmNew::DetSet<SiPixelCluster>::const_iterator  di;
    for(di = isearch->begin(); di != isearch->end(); di++) {
      numberOfClusters++;
      if(endcap) numberOfFpixClusters++;
      float charge = 0.001*(di->charge()); // total charge of cluster
      float x = di->x();                   // barycenter x position
      float y = di->y();                   // barycenter y position
      int size = di->size();               // total size of cluster (in pixels)
      int sizeX = di->sizeX();             // size of cluster in x-direction
      int sizeY = di->sizeY();             // size of cluster in y-direction
      int minPixelRow = di->minPixelRow(); // min x index
      int maxPixelRow = di->maxPixelRow(); // max x index
      int minPixelCol = di->minPixelCol(); // min y index
      int maxPixelCol = di->maxPixelCol(); // max y index

      const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(DetId(id_)) );

      const PixelTopology * topol = &(theGeomDet->specificTopology());
      LocalPoint clustlp = topol->localPosition( MeasurementPoint(x, y) );
      GlobalPoint clustgp = theGeomDet->surface().toGlobal( clustlp );

      if(barrel){
   for (std::vector<MonitorElement*>::iterator i = layers.begin(); i != layers.end(); i++)
   {
     (*i)->Fill(clustgp.z(),clustgp.phi());
   }
      }else if(endcap){
	if(clustgp.z()>0){
     for (std::vector<MonitorElement*>::iterator i = diskspz.begin(); i != diskspz.end(); i++)
     {
       (*i)->Fill(clustgp.x(),clustgp.y());
     }
	}else{
     for (std::vector<MonitorElement*>::iterator i = disksmz.begin(); i != disksmz.end(); i++)
     {
       (*i)->Fill(clustgp.x(),clustgp.y());
     }
	} 
      }
      if(!reducedSet)
	{
	  (meMinRow_)->Fill((int)minPixelRow);
	  (meMaxRow_)->Fill((int)maxPixelRow);
	  (meMinCol_)->Fill((int)minPixelCol);
	  (meMaxCol_)->Fill((int)maxPixelCol);
	  (meSizeX_)->Fill((int)sizeX);
	  (meSizeY_)->Fill((int)sizeY);
 	  (meX_)->Fill((float)x);
	  (meY_)->Fill((float)y);
	  if(twoD)(mePixClusters_)->Fill((float)y,(float)x);
	  else{
	  	  (mePixClusters_px_)->Fill((float)y);
	        (mePixClusters_py_)->Fill((float)x);
	  }
	}

      if(barrel && smileyon){
        (meSizeYvsEtaBarrel_)->Fill(clustgp.eta(),sizeY);
	//std::cout << "Cluster Global x y z theta eta " << clustgp.x() << " " << clustgp.y() << " " << clustgp.z() << " " << clustgp.theta() << " " << clustgp.eta() << std::endl;
      }      
      if(ladon && barrel){
	(meChargeLad_)->Fill((float)charge);
	(meSizeLad_)->Fill((int)size);
	if(!reducedSet)
	{
	(meMinRowLad_)->Fill((int)minPixelRow);
	(meMaxRowLad_)->Fill((int)maxPixelRow);
	(meMinColLad_)->Fill((int)minPixelCol);
	(meMaxColLad_)->Fill((int)maxPixelCol);
	(meXLad_)->Fill((float)x);
	(meYLad_)->Fill((float)y);
	(meSizeXLad_)->Fill((int)sizeX);
	(meSizeYLad_)->Fill((int)sizeY);
	if(twoD) (mePixClustersLad_)->Fill((float)y,(float)x);
	else{
	  (mePixClustersLad_px_)->Fill((float)y);
	  (mePixClustersLad_py_)->Fill((float)x);
	}
	}
      }
      if(layon && barrel){
	(meChargeLay_)->Fill((float)charge);
	(meSizeLay_)->Fill((int)size);
	if(!reducedSet)
	{
	(meMinRowLay_)->Fill((int)minPixelRow);
	(meMaxRowLay_)->Fill((int)maxPixelRow);
	(meMinColLay_)->Fill((int)minPixelCol);
	(meMaxColLay_)->Fill((int)maxPixelCol);
	(meXLay_)->Fill((float)x);
	(meYLay_)->Fill((float)y);
	(meSizeXLay_)->Fill((int)sizeX);
	(meSizeYLay_)->Fill((int)sizeY);
	if(twoD) (mePixClustersLay_)->Fill((float)y,(float)x);
	else{
	  (mePixClustersLay_px_)->Fill((float)y);
	  (mePixClustersLay_py_)->Fill((float)x);
	}
	}
      }
      if(phion && barrel){
	(meChargePhi_)->Fill((float)charge);
	(meSizePhi_)->Fill((int)size);
	if(!reducedSet)
	{
	(meMinRowPhi_)->Fill((int)minPixelRow);
	(meMaxRowPhi_)->Fill((int)maxPixelRow);
	(meMinColPhi_)->Fill((int)minPixelCol);
	(meMaxColPhi_)->Fill((int)maxPixelCol);
	(meXPhi_)->Fill((float)x);
	(meYPhi_)->Fill((float)y);
	(meSizeXPhi_)->Fill((int)sizeX);
	(meSizeYPhi_)->Fill((int)sizeY);
	if(twoD) (mePixClustersPhi_)->Fill((float)y,(float)x);
	else{
	  (mePixClustersPhi_px_)->Fill((float)y);
	  (mePixClustersPhi_py_)->Fill((float)x);
	}
	}
      }
      if(bladeon && endcap){
	(meChargeBlade_)->Fill((float)charge);
	(meSizeBlade_)->Fill((int)size);
	if(!reducedSet)
	{
	(meMinRowBlade_)->Fill((int)minPixelRow);
	(meMaxRowBlade_)->Fill((int)maxPixelRow);
	(meMinColBlade_)->Fill((int)minPixelCol);
	(meMaxColBlade_)->Fill((int)maxPixelCol);
	(meXBlade_)->Fill((float)x);
	(meYBlade_)->Fill((float)y);
	(meSizeXBlade_)->Fill((int)sizeX);
	(meSizeYBlade_)->Fill((int)sizeY);
	}
      }
      if(diskon && endcap){
	(meChargeDisk_)->Fill((float)charge);
	(meSizeDisk_)->Fill((int)size);
	if(!reducedSet)
	{
	(meMinRowDisk_)->Fill((int)minPixelRow);
	(meMaxRowDisk_)->Fill((int)maxPixelRow);
	(meMinColDisk_)->Fill((int)minPixelCol);
	(meMaxColDisk_)->Fill((int)maxPixelCol);
	(meXDisk_)->Fill((float)x);
	(meYDisk_)->Fill((float)y);
	(meSizeXDisk_)->Fill((int)sizeX);
	(meSizeYDisk_)->Fill((int)sizeY);
	}
      }
      
      if(ringon && endcap){
	(meChargeRing_)->Fill((float)charge);
	(meSizeRing_)->Fill((int)size);
	if(!reducedSet)
	{
	(meMinRowRing_)->Fill((int)minPixelRow);
	(meMaxRowRing_)->Fill((int)maxPixelRow);
	(meMinColRing_)->Fill((int)minPixelCol);
	(meMaxColRing_)->Fill((int)maxPixelCol);
	(meXRing_)->Fill((float)x);
	(meYRing_)->Fill((float)y);
	(meSizeXRing_)->Fill((int)sizeX);
	(meSizeYRing_)->Fill((int)sizeY);
	if(twoD) (mePixClustersRing_)->Fill((float)y,(float)x);
	else{
	  (mePixClustersRing_px_)->Fill((float)y);
	  (mePixClustersRing_py_)->Fill((float)x);
	}
	}
      }
    }
    //if(modon) (meNClusters_)->Fill((float)numberOfClusters);
    if(ladon && barrel) (meNClustersLad_)->Fill((float)numberOfClusters);
    if(layon && barrel) (meNClustersLay_)->Fill((float)numberOfClusters);
    if(phion && barrel) (meNClustersPhi_)->Fill((float)numberOfClusters);
    if(bladeon && endcap) (meNClustersBlade_)->Fill((float)numberOfClusters);
    if(diskon && endcap) (meNClustersDisk_)->Fill((float)numberOfClusters);
    if(ringon && endcap) (meNClustersRing_)->Fill((float)numberOfClusters);

    //std::cout<<"number of clusters="<<numberOfClusters<<std::endl;
      

  }
  
  
  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;
  return numberOfFpixClusters;
  
}
