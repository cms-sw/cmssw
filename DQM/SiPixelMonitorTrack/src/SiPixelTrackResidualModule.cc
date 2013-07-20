// Package:    SiPixelMonitorTrack
// Class:      SiPixelTrackResidualModule
// 
// class SiPixelTrackResidualModule SiPixelTrackResidualModule.cc 
//       DQM/SiPixelMonitorTrack/src/SiPixelTrackResidualModule.cc
//
// Description: SiPixel hit-to-track residual data quality monitoring modules
// Implementation: prototype -> improved -> never final - end of the 1st step 
//
// Original Author: Shan-Huei Chuang
//         Created: Fri Mar 23 18:41:42 CET 2007
// $Id: SiPixelTrackResidualModule.cc,v 1.10 2012/09/11 09:37:41 clseitz Exp $


#include <string>
#include <iostream>

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
#include "DQM/SiPixelMonitorTrack/interface/SiPixelTrackResidualModule.h"

// Data Formats
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"


using namespace std; 


SiPixelTrackResidualModule::SiPixelTrackResidualModule() : id_(0) {
  bBookTracks = true;
}


SiPixelTrackResidualModule::SiPixelTrackResidualModule(uint32_t id) : id_(id) { 
  bBookTracks = true;
}


SiPixelTrackResidualModule::~SiPixelTrackResidualModule() { 
 
}


void SiPixelTrackResidualModule::book(const edm::ParameterSet& iConfig, bool reducedSet, int type) {
  DQMStore* dbe = edm::Service<DQMStore>().operator->();

  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  bool isHalfModule = false;
  if(barrel){
    isHalfModule = PixelBarrelName(DetId(id_)).isHalfModule(); 
  }
  
  edm::InputTag src = iConfig.getParameter<edm::InputTag>("src");
  std::string hisID;

  if(type==0){
    SiPixelHistogramId* theHistogramId = new SiPixelHistogramId(src.label());
    hisID = theHistogramId->setHistoId("residualX",id_);
    meResidualX_ = dbe->book1D(hisID,"Hit-to-Track Residual in r-phi",100,-150,150);
    meResidualX_->setAxisTitle("hit-to-track residual in r-phi (um)",1);
    hisID = theHistogramId->setHistoId("residualY",id_);
    meResidualY_ = dbe->book1D(hisID,"Hit-to-Track Residual in Z",100,-300,300);
    meResidualY_->setAxisTitle("hit-to-track residual in z (um)",1);
    // Number of clusters
    hisID = theHistogramId->setHistoId("nclusters_OnTrack",id_);
    meNClusters_onTrack_ = dbe->book1D(hisID,"Number of Clusters (on Track)",10,0.,10.);
    meNClusters_onTrack_->setAxisTitle("Number of Clusters on Track",1);
    // Total cluster charge in ke
    hisID = theHistogramId->setHistoId("charge_OnTrack",id_);
    meCharge_onTrack_ = dbe->book1D(hisID,"Normalized Cluster charge (on Track)",100,0.,200.);
    meCharge_onTrack_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    hisID = theHistogramId->setHistoId("size_OnTrack",id_);
    meSize_onTrack_ = dbe->book1D(hisID,"Total cluster size (on Track)",30,0.,30.);
    meSize_onTrack_->setAxisTitle("Cluster size [number of pixels]",1);
    // Number of clusters
    hisID = theHistogramId->setHistoId("nclusters_OffTrack",id_);
    meNClusters_offTrack_ = dbe->book1D(hisID,"Number of Clusters (off Track)",35,0.,35.);
    meNClusters_offTrack_->setAxisTitle("Number of Clusters off Track",1);
    // Total cluster charge in ke
    hisID = theHistogramId->setHistoId("charge_OffTrack",id_);
    meCharge_offTrack_ = dbe->book1D(hisID,"Cluster charge (off Track)",100,0.,200.);
    meCharge_offTrack_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    hisID = theHistogramId->setHistoId("size_OffTrack",id_);
    meSize_offTrack_ = dbe->book1D(hisID,"Total cluster size (off Track)",30,0.,30.);
    meSize_offTrack_->setAxisTitle("Cluster size [number of pixels]",1);
    if(!reducedSet){
      // Cluster width on the x-axis
      hisID = theHistogramId->setHistoId("sizeX_OnTrack",id_);
      meSizeX_onTrack_ = dbe->book1D(hisID,"Cluster x-width (rows) (on Track)",10,0.,10.);
      meSizeX_onTrack_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      hisID = theHistogramId->setHistoId("sizeY_OnTrack",id_);
      meSizeY_onTrack_ = dbe->book1D(hisID,"Cluster y-width (columns) (on Track)",15,0.,15.);
      meSizeY_onTrack_->setAxisTitle("Cluster y-size [columns]",1);
      // Cluster width on the x-axis
      hisID = theHistogramId->setHistoId("sizeX_OffTrack",id_);
      meSizeX_offTrack_ = dbe->book1D(hisID,"Cluster x-width (rows) (off Track)",10,0.,10.);
      meSizeX_offTrack_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      hisID = theHistogramId->setHistoId("sizeY_OffTrack",id_);
      meSizeY_offTrack_ = dbe->book1D(hisID,"Cluster y-width (columns) (off Track)",15,0.,15.);
      meSizeY_offTrack_->setAxisTitle("Cluster y-size [columns]",1);
    }
    delete theHistogramId;
  }

  if(type==1 && barrel){
    uint32_t DBladder = PixelBarrelName(DetId(id_)).ladderName();
    char sladder[80]; sprintf(sladder,"Ladder_%02i",DBladder);
    hisID = src.label() + "_" + sladder;
    if(isHalfModule) hisID += "H";
    else hisID += "F";
    meResidualXLad_ = dbe->book1D("residualX_"+hisID,"Hit-to-Track Residual in r-phi",100,-150,150);
    meResidualXLad_->setAxisTitle("hit-to-track residual in r-phi (um)",1);
    meResidualYLad_ = dbe->book1D("residualY_"+hisID,"Hit-to-Track Residual in Z",100,-300,300);
    meResidualYLad_->setAxisTitle("hit-to-track residual in z (um)",1);
    // Number of clusters
    meNClusters_onTrackLad_ = dbe->book1D("nclusters_OnTrack_" + hisID,"Number of Clusters (on Track)",10,0.,10.);
    meNClusters_onTrackLad_->setAxisTitle("Number of Clusters on Track",1);
    // Total cluster charge in MeV
    meCharge_onTrackLad_ = dbe->book1D("charge_OnTrack_" + hisID,"Normalized Cluster charge (on Track)",100,0.,200.);
    meCharge_onTrackLad_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSize_onTrackLad_ = dbe->book1D("size_OnTrack_" + hisID,"Total cluster size (on Track)",30,0.,30.);
    meSize_onTrackLad_->setAxisTitle("Cluster size [number of pixels]",1);
    // Number of clusters
    meNClusters_offTrackLad_ = dbe->book1D("nclusters_OffTrack_" + hisID,"Number of Clusters (off Track)",35,0.,35.);
    meNClusters_offTrackLad_->setAxisTitle("Number of Clusters off Track",1);
    // Total cluster charge in MeV
    meCharge_offTrackLad_ = dbe->book1D("charge_OffTrack_" + hisID,"Cluster charge (off Track)",100,0.,200.);
    meCharge_offTrackLad_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSize_offTrackLad_ = dbe->book1D("size_OffTrack_" + hisID,"Total cluster size (off Track)",30,0.,30.);
    meSize_offTrackLad_->setAxisTitle("Cluster size [number of pixels]",1);
    if(!reducedSet){
      // Cluster width on the x-axis
      meSizeX_offTrackLad_ = dbe->book1D("sizeX_OffTrack_" + hisID,"Cluster x-width (rows) (off Track)",10,0.,10.);
      meSizeX_offTrackLad_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeY_offTrackLad_ = dbe->book1D("sizeY_OffTrack_" + hisID,"Cluster y-width (columns) (off Track)",15,0.,15.);
      meSizeY_offTrackLad_->setAxisTitle("Cluster y-size [columns]",1);
      // Cluster width on the x-axis
      meSizeX_onTrackLad_ = dbe->book1D("sizeX_OnTrack_" + hisID,"Cluster x-width (rows) (on Track)",10,0.,10.);
      meSizeX_onTrackLad_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeY_onTrackLad_ = dbe->book1D("sizeY_OnTrack_" + hisID,"Cluster y-width (columns) (on Track)",15,0.,15.);
      meSizeY_onTrackLad_->setAxisTitle("Cluster y-size [columns]",1);
    }
  }

  if(type==2 && barrel){
    uint32_t DBlayer = PixelBarrelName(DetId(id_)).layerName();
    char slayer[80]; sprintf(slayer,"Layer_%i",DBlayer);
    hisID = src.label() + "_" + slayer;
    meResidualXLay_ = dbe->book1D("residualX_"+hisID,"Hit-to-Track Residual in r-phi",100,-150,150);
    meResidualXLay_->setAxisTitle("hit-to-track residual in r-phi (um)",1);
    meResidualYLay_ = dbe->book1D("residualY_"+hisID,"Hit-to-Track Residual in Z",100,-300,300);
    meResidualYLay_->setAxisTitle("hit-to-track residual in z (um)",1);
    // Number of clusters
    meNClusters_onTrackLay_ = dbe->book1D("nclusters_OnTrack_" + hisID,"Number of Clusters (on Track)",10,0.,10.);
    meNClusters_onTrackLay_->setAxisTitle("Number of Clusters on Track",1);
    // Total cluster charge in MeV
    meCharge_onTrackLay_ = dbe->book1D("charge_OnTrack_" + hisID,"Normalized Cluster charge (on Track)",100,0.,200.);
    meCharge_onTrackLay_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSize_onTrackLay_ = dbe->book1D("size_OnTrack_" + hisID,"Total cluster size (on Track)",30,0.,30.);
    meSize_onTrackLay_->setAxisTitle("Cluster size [number of pixels]",1);    
    // Number of clusters
    meNClusters_offTrackLay_ = dbe->book1D("nclusters_OffTrack_" + hisID,"Number of Clusters (off Track)",35,0.,35.);
    meNClusters_offTrackLay_->setAxisTitle("Number of Clusters off Track",1);
    // Total cluster charge in MeV
    meCharge_offTrackLay_ = dbe->book1D("charge_OffTrack_" + hisID,"Cluster charge (off Track)",100,0.,200.);
    meCharge_offTrackLay_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSize_offTrackLay_ = dbe->book1D("size_OffTrack_" + hisID,"Total cluster size (off Track)",30,0.,30.);
    meSize_offTrackLay_->setAxisTitle("Cluster size [number of pixels]",1);
    if(!reducedSet){
      // Cluster width on the x-axis
      meSizeX_onTrackLay_ = dbe->book1D("sizeX_OnTrack_" + hisID,"Cluster x-width (rows) (on Track)",10,0.,10.);
      meSizeX_onTrackLay_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeY_onTrackLay_ = dbe->book1D("sizeY_OnTrack_" + hisID,"Cluster y-width (columns) (on Track)",15,0.,15.);
      meSizeY_onTrackLay_->setAxisTitle("Cluster y-size [columns]",1);
      // Cluster width on the x-axis
      meSizeX_offTrackLay_ = dbe->book1D("sizeX_OffTrack_" + hisID,"Cluster x-width (rows) (off Track)",10,0.,10.);
      meSizeX_offTrackLay_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeY_offTrackLay_ = dbe->book1D("sizeY_OffTrack_" + hisID,"Cluster y-width (columns) (off Track)",15,0.,15.);
      meSizeY_offTrackLay_->setAxisTitle("Cluster y-size [columns]",1);
    }
  }

  if(type==3 && barrel){
    uint32_t DBmodule = PixelBarrelName(DetId(id_)).moduleName();
    char smodule[80]; sprintf(smodule,"Ring_%i",DBmodule);
    hisID = src.label() + "_" + smodule;
    meResidualXPhi_ = dbe->book1D("residualX_"+hisID,"Hit-to-Track Residual in r-phi",100,-150,150);
    meResidualXPhi_->setAxisTitle("hit-to-track residual in r-phi (um)",1);
    meResidualYPhi_ = dbe->book1D("residualY_"+hisID,"Hit-to-Track Residual in Z",100,-300,300);
    meResidualYPhi_->setAxisTitle("hit-to-track residual in z (um)",1);
    // Number of clusters
    meNClusters_onTrackPhi_ = dbe->book1D("nclusters_OnTrack_" + hisID,"Number of Clusters (on Track)",10,0.,10.);
    meNClusters_onTrackPhi_->setAxisTitle("Number of Clusters on Track",1);
    // Total cluster charge in MeV
    meCharge_onTrackPhi_ = dbe->book1D("charge_OnTrack_" + hisID,"Normalized Cluster charge (on Track)",100,0.,200.);
    meCharge_onTrackPhi_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSize_onTrackPhi_ = dbe->book1D("size_OnTrack_" + hisID,"Total cluster size (on Track)",30,0.,30.);
    meSize_onTrackPhi_->setAxisTitle("Cluster size [number of pixels]",1);    
    // Number of clusters
    meNClusters_offTrackPhi_ = dbe->book1D("nclusters_OffTrack_" + hisID,"Number of Clusters (off Track)",35,0.,35.);
    meNClusters_offTrackPhi_->setAxisTitle("Number of Clusters off Track",1);
    // Total cluster charge in MeV
    meCharge_offTrackPhi_ = dbe->book1D("charge_OffTrack_" + hisID,"Cluster charge (off Track)",100,0.,200.);
    meCharge_offTrackPhi_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSize_offTrackPhi_ = dbe->book1D("size_OffTrack_" + hisID,"Total cluster size (off Track)",30,0.,30.);
    meSize_offTrackPhi_->setAxisTitle("Cluster size [number of pixels]",1);
    if(!reducedSet){
      // Cluster width on the x-axis
      meSizeX_onTrackPhi_ = dbe->book1D("sizeX_OnTrack_" + hisID,"Cluster x-width (rows) (on Track)",10,0.,10.);
      meSizeX_onTrackPhi_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeY_onTrackPhi_ = dbe->book1D("sizeY_OnTrack_" + hisID,"Cluster y-width (columns) (on Track)",15,0.,15.);
      meSizeY_onTrackPhi_->setAxisTitle("Cluster y-size [columns]",1);
      // Cluster width on the x-axis
      meSizeX_offTrackPhi_ = dbe->book1D("sizeX_OffTrack_" + hisID,"Cluster x-width (rows) (off Track)",10,0.,10.);
      meSizeX_offTrackPhi_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeY_offTrackPhi_ = dbe->book1D("sizeY_OffTrack_" + hisID,"Cluster y-width (columns) (off Track)",15,0.,15.);
      meSizeY_offTrackPhi_->setAxisTitle("Cluster y-size [columns]",1);
    }
  }

  if(type==4 && endcap){
    uint32_t blade= PixelEndcapName(DetId(id_)).bladeName();
    char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
    hisID = src.label() + "_" + sblade;
    meResidualXBlade_ = dbe->book1D("residualX_"+hisID,"Hit-to-Track Residual in r-phi",100,-150,150);
    meResidualXBlade_->setAxisTitle("hit-to-track residual in r-phi (um)",1);
    meResidualYBlade_ = dbe->book1D("residualY_"+hisID,"Hit-to-Track Residual in Z",100,-300,300);
    meResidualYBlade_->setAxisTitle("hit-to-track residual in z (um)",1);
    // Number of clusters
    meNClusters_onTrackBlade_ = dbe->book1D("nclusters_OnTrack_" + hisID,"Number of Clusters (on Track)",10,0.,10.);
    meNClusters_onTrackBlade_->setAxisTitle("Number of Clusters on Track",1);
    // Total cluster charge in MeV
    meCharge_onTrackBlade_ = dbe->book1D("charge_OnTrack_" + hisID,"Normalized Cluster charge (on Track)",100,0.,200.);
    meCharge_onTrackBlade_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSize_onTrackBlade_ = dbe->book1D("size_OnTrack_" + hisID,"Total cluster size (on Track)",30,0.,30.);
    meSize_onTrackBlade_->setAxisTitle("Cluster size [number of pixels]",1);    
    // Number of clusters
    meNClusters_offTrackBlade_ = dbe->book1D("nclusters_OffTrack_" + hisID,"Number of Clusters (off Track)",35,0.,35.);
    meNClusters_offTrackBlade_->setAxisTitle("Number of Clusters off Track",1);
    // Total cluster charge in MeV
    meCharge_offTrackBlade_ = dbe->book1D("charge_OffTrack_" + hisID,"Cluster charge (off Track)",100,0.,200.);
    meCharge_offTrackBlade_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSize_offTrackBlade_ = dbe->book1D("size_OffTrack_" + hisID,"Total cluster size (off Track)",30,0.,30.);
    meSize_offTrackBlade_->setAxisTitle("Cluster size [number of pixels]",1);
    if(!reducedSet){
      // Cluster width on the x-axis
      meSizeX_onTrackBlade_ = dbe->book1D("sizeX_OnTrack_" + hisID,"Cluster x-width (rows) (on Track)",10,0.,10.);
      meSizeX_onTrackBlade_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeY_onTrackBlade_ = dbe->book1D("sizeY_OnTrack_" + hisID,"Cluster y-width (columns) (on Track)",15,0.,15.);
      meSizeY_onTrackBlade_->setAxisTitle("Cluster y-size [columns]",1);
      // Cluster width on the x-axis
      meSizeX_offTrackBlade_ = dbe->book1D("sizeX_OffTrack_" + hisID,"Cluster x-width (rows) (off Track)",10,0.,10.);
      meSizeX_offTrackBlade_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeY_offTrackBlade_ = dbe->book1D("sizeY_OffTrack_" + hisID,"Cluster y-width (columns) (off Track)",15,0.,15.);
      meSizeY_offTrackBlade_->setAxisTitle("Cluster y-size [columns]",1);
    }
  }

  if(type==5 && endcap){
    uint32_t disk = PixelEndcapName(DetId(id_)).diskName();
    char sdisk[80]; sprintf(sdisk, "Disk_%i",disk);
    hisID = src.label() + "_" + sdisk;
    meResidualXDisk_ = dbe->book1D("residualX_"+hisID,"Hit-to-Track Residual in r-phi",100,-150,150);
    meResidualXDisk_->setAxisTitle("hit-to-track residual in r-phi (um)",1);
    meResidualYDisk_ = dbe->book1D("residualY_"+hisID,"Hit-to-Track Residual in Z",100,-300,300);
    meResidualYDisk_->setAxisTitle("hit-to-track residual in z (um)",1);
    // Number of clusters
    meNClusters_onTrackDisk_ = dbe->book1D("nclusters_OnTrack_" + hisID,"Number of Clusters (on Track)",10,0.,10.);
    meNClusters_onTrackDisk_->setAxisTitle("Number of Clusters on Track",1);
    // Total cluster charge in MeV
    meCharge_onTrackDisk_ = dbe->book1D("charge_OnTrack_" + hisID,"Normalized Cluster charge (on Track)",100,0.,200.);
    meCharge_onTrackDisk_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSize_onTrackDisk_ = dbe->book1D("size_OnTrack_" + hisID,"Total cluster size (on Track)",30,0.,30.);
    meSize_onTrackDisk_->setAxisTitle("Cluster size [number of pixels]",1);    
    // Number of clusters
    meNClusters_offTrackDisk_ = dbe->book1D("nclusters_OffTrack_" + hisID,"Number of Clusters (off Track)",35,0.,35.);
    meNClusters_offTrackDisk_->setAxisTitle("Number of Clusters off Track",1);
    // Total cluster charge in MeV
    meCharge_offTrackDisk_ = dbe->book1D("charge_OffTrack_" + hisID,"Cluster charge (off Track)",100,0.,200.);
    meCharge_offTrackDisk_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSize_offTrackDisk_ = dbe->book1D("size_OffTrack_" + hisID,"Total cluster size (off Track)",30,0.,30.);
    meSize_offTrackDisk_->setAxisTitle("Cluster size [number of pixels]",1);
    if(!reducedSet){
      // Cluster width on the x-axis
      meSizeX_onTrackDisk_ = dbe->book1D("sizeX_OnTrack_" + hisID,"Cluster x-width (rows) (on Track)",10,0.,10.);
      meSizeX_onTrackDisk_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeY_onTrackDisk_ = dbe->book1D("sizeY_OnTrack_" + hisID,"Cluster y-width (columns) (on Track)",15,0.,15.);
      meSizeY_onTrackDisk_->setAxisTitle("Cluster y-size [columns]",1);
      // Cluster width on the x-axis
      meSizeX_offTrackDisk_ = dbe->book1D("sizeX_OffTrack_" + hisID,"Cluster x-width (rows) (off Track)",10,0.,10.);
      meSizeX_offTrackDisk_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeY_offTrackDisk_ = dbe->book1D("sizeY_OffTrack_" + hisID,"Cluster y-width (columns) (off Track)",15,0.,15.);
      meSizeY_offTrackDisk_->setAxisTitle("Cluster y-size [columns]",1);
    }
  }

  if(type==6 && endcap){
    uint32_t panel= PixelEndcapName(DetId(id_)).pannelName();
    uint32_t module= PixelEndcapName(DetId(id_)).plaquetteName();
    char slab[80]; sprintf(slab, "Panel_%i_Ring_%i",panel, module);
    hisID = src.label() + "_" + slab;
    meResidualXRing_ = dbe->book1D("residualX_"+hisID,"Hit-to-Track Residual in r-phi",100,-150,150);
    meResidualXRing_->setAxisTitle("hit-to-track residual in r-phi (um)",1);
    meResidualYRing_ = dbe->book1D("residualY_"+hisID,"Hit-to-Track Residual in Z",100,-300,300);
    meResidualYRing_->setAxisTitle("hit-to-track residual in z (um)",1);
    // Number of clusters
    meNClusters_onTrackRing_ = dbe->book1D("nclusters_OnTrack_" + hisID,"Number of Clusters (on Track)",10,0.,10.);
    meNClusters_onTrackRing_->setAxisTitle("Number of Clusters on Track",1);
    // Total cluster charge in MeV
    meCharge_onTrackRing_ = dbe->book1D("charge_OnTrack_" + hisID,"Normalized Cluster charge (on Track)",100,0.,200.);
    meCharge_onTrackRing_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSize_onTrackRing_ = dbe->book1D("size_OnTrack_" + hisID,"Total cluster size (on Track)",30,0.,30.);
    meSize_onTrackRing_->setAxisTitle("Cluster size [number of pixels]",1);    
    // Number of clusters
    meNClusters_offTrackRing_ = dbe->book1D("nclusters_OffTrack_" + hisID,"Number of Clusters (off Track)",35,0.,35.);
    meNClusters_offTrackRing_->setAxisTitle("Number of Clusters off Track",1);
    // Total cluster charge in MeV
    meCharge_offTrackRing_ = dbe->book1D("charge_OffTrack_" + hisID,"Cluster charge (off Track)",100,0.,200.);
    meCharge_offTrackRing_->setAxisTitle("Charge [kilo electrons]",1);
    // Total cluster size (in pixels)
    meSize_offTrackRing_ = dbe->book1D("size_OffTrack_" + hisID,"Total cluster size (off Track)",30,0.,30.);
    meSize_offTrackRing_->setAxisTitle("Cluster size [number of pixels]",1);
    if(!reducedSet){
      // Cluster width on the x-axis
      meSizeX_onTrackRing_ = dbe->book1D("sizeX_OnTrack_" + hisID,"Cluster x-width (rows) (on Track)",10,0.,10.);
      meSizeX_onTrackRing_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeY_onTrackRing_ = dbe->book1D("sizeY_OnTrack_" + hisID,"Cluster y-width (columns) (on Track)",15,0.,15.);
      meSizeY_onTrackRing_->setAxisTitle("Cluster y-size [columns]",1);
      // Cluster width on the x-axis
      meSizeX_offTrackRing_ = dbe->book1D("sizeX_OffTrack_" + hisID,"Cluster x-width (rows) (off Track)",10,0.,10.);
      meSizeX_offTrackRing_->setAxisTitle("Cluster x-size [rows]",1);
      // Cluster width on the y-axis
      meSizeY_offTrackRing_ = dbe->book1D("sizeY_OffTrack_" + hisID,"Cluster y-width (columns) (off Track)",15,0.,15.);
      meSizeY_offTrackRing_->setAxisTitle("Cluster y-size [columns]",1);
    }
  }
}


void SiPixelTrackResidualModule::fill(const Measurement2DVector& residual, bool reducedSet, bool modon, bool ladon, bool layon, bool phion, bool bladeon, bool diskon, bool ringon) {

  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);

  if(modon){
    (meResidualX_)->Fill(residual.x()); 
    (meResidualY_)->Fill(residual.y()); 
  }

  if(ladon && barrel){
    (meResidualXLad_)->Fill(residual.x()); 
    (meResidualYLad_)->Fill(residual.y()); 
  }

  if(layon && barrel){
    (meResidualXLay_)->Fill(residual.x()); 
    (meResidualYLay_)->Fill(residual.y()); 
  }
  if(phion && barrel){
    (meResidualXPhi_)->Fill(residual.x()); 
    (meResidualYPhi_)->Fill(residual.y()); 
  }

  if(bladeon && endcap){
    (meResidualXBlade_)->Fill(residual.x()); 
    (meResidualYBlade_)->Fill(residual.y()); 
  }

  if(diskon && endcap){
    (meResidualXDisk_)->Fill(residual.x()); 
    (meResidualYDisk_)->Fill(residual.y()); 
  }

  if(ringon && endcap){
    (meResidualXRing_)->Fill(residual.x()); 
    (meResidualYRing_)->Fill(residual.y()); 
  }
}


void SiPixelTrackResidualModule::fill(const SiPixelCluster &clust, bool onTrack, double corrCharge, bool reducedSet, bool modon, bool ladon, bool layon, bool phion, bool bladeon, bool diskon, bool ringon){

  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);

  float charge = 0.001*(clust.charge()); // total charge of cluster
  if(onTrack) charge = corrCharge;              // corrected cluster charge
  int size = clust.size();               // total size of cluster (in pixels)
  int sizeX = clust.sizeX();             // size of cluster in x-clustrection
  int sizeY = clust.sizeY();             // size of cluster in y-direction
  
  if(onTrack){
    if(modon){
      (meCharge_onTrack_)->Fill((float)charge);
      (meSize_onTrack_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_onTrack_)->Fill((int)sizeX);
        (meSizeY_onTrack_)->Fill((int)sizeY);
      }
    }
    if(barrel && ladon){
      (meCharge_onTrackLad_)->Fill((float)charge);
      (meSize_onTrackLad_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_onTrackLad_)->Fill((int)sizeX);
        (meSizeY_onTrackLad_)->Fill((int)sizeY);
      }
    }
    if(barrel && layon){
      (meCharge_onTrackLay_)->Fill((float)charge);
      (meSize_onTrackLay_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_onTrackLay_)->Fill((int)sizeX);
        (meSizeY_onTrackLay_)->Fill((int)sizeY);
      }
    }   
    if(barrel && phion){
      (meCharge_onTrackPhi_)->Fill((float)charge);
      (meSize_onTrackPhi_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_onTrackPhi_)->Fill((int)sizeX);
        (meSizeY_onTrackPhi_)->Fill((int)sizeY);
      }
    }
    if(endcap && bladeon){
      (meCharge_onTrackBlade_)->Fill((float)charge);
      (meSize_onTrackBlade_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_onTrackBlade_)->Fill((int)sizeX);
        (meSizeY_onTrackBlade_)->Fill((int)sizeY);
      }
    }
    if(endcap && diskon){
      (meCharge_onTrackDisk_)->Fill((float)charge);
      (meSize_onTrackDisk_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_onTrackDisk_)->Fill((int)sizeX);
        (meSizeY_onTrackDisk_)->Fill((int)sizeY);
      }
    }
    if(endcap && ringon){
      (meCharge_onTrackRing_)->Fill((float)charge);
      (meSize_onTrackRing_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_onTrackRing_)->Fill((int)sizeX);
        (meSizeY_onTrackRing_)->Fill((int)sizeY);
      }
    }
  }

  if(!onTrack){
    if(modon){
      (meCharge_offTrack_)->Fill((float)charge);
      (meSize_offTrack_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_offTrack_)->Fill((int)sizeX);
        (meSizeY_offTrack_)->Fill((int)sizeY);
      }
    }
    if(barrel && ladon){
      (meCharge_offTrackLad_)->Fill((float)charge);
      (meSize_offTrackLad_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_offTrackLad_)->Fill((int)sizeX);
        (meSizeY_offTrackLad_)->Fill((int)sizeY);
      }
    }
    if(barrel && layon){
      (meCharge_offTrackLay_)->Fill((float)charge);
      (meSize_offTrackLay_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_offTrackLay_)->Fill((int)sizeX);
        (meSizeY_offTrackLay_)->Fill((int)sizeY);
      }
    }   
    if(barrel && phion){
      (meCharge_offTrackPhi_)->Fill((float)charge);
      (meSize_offTrackPhi_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_offTrackPhi_)->Fill((int)sizeX);
        (meSizeY_offTrackPhi_)->Fill((int)sizeY);
      }
    }
    if(endcap && bladeon){
      (meCharge_offTrackBlade_)->Fill((float)charge);
      (meSize_offTrackBlade_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_offTrackBlade_)->Fill((int)sizeX);
        (meSizeY_offTrackBlade_)->Fill((int)sizeY);
      }
    }
    if(endcap && diskon){
      (meCharge_offTrackDisk_)->Fill((float)charge);
      (meSize_offTrackDisk_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_offTrackDisk_)->Fill((int)sizeX);
        (meSizeY_offTrackDisk_)->Fill((int)sizeY);
      }
    }
    if(endcap && ringon){
      (meCharge_offTrackRing_)->Fill((float)charge);
      (meSize_offTrackRing_)->Fill((int)size);
      if(!reducedSet){
        (meSizeX_offTrackRing_)->Fill((int)sizeX);
        (meSizeY_offTrackRing_)->Fill((int)sizeY);
      }
    }
  }


  
}

void SiPixelTrackResidualModule::nfill(int onTrack, int offTrack, bool reducedSet, bool modon, bool ladon, bool layon, bool phion, bool bladeon, bool diskon, bool ringon){

  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  bool fillOn = false; if(onTrack>0) fillOn = true; 
  bool fillOff = false; if(offTrack>0) fillOff = true; 

  if(modon){
    if(fillOn) meNClusters_onTrack_->Fill(onTrack); 
    if(fillOff) meNClusters_offTrack_->Fill(offTrack); 
  }
  if(ladon && barrel){
    if(fillOn) meNClusters_onTrackLad_->Fill(onTrack); 
    if(fillOff) meNClusters_offTrackLad_->Fill(offTrack); 
  }
  
  if(layon && barrel){
    if(fillOn) meNClusters_onTrackLay_->Fill(onTrack); 
    if(fillOff) meNClusters_offTrackLay_->Fill(offTrack); 
  }
  if(phion && barrel){
    if(fillOn) meNClusters_onTrackPhi_->Fill(onTrack); 
    if(fillOff) meNClusters_offTrackPhi_->Fill(offTrack); 
  }
  if(bladeon && endcap){
    if(fillOn) meNClusters_onTrackBlade_->Fill(onTrack); 
    if(fillOff) meNClusters_offTrackBlade_->Fill(offTrack); 
  }
  if(diskon && endcap){
    if(fillOn) meNClusters_onTrackDisk_->Fill(onTrack); 
    if(fillOff) meNClusters_offTrackDisk_->Fill(offTrack); 
  }
  if(ringon && endcap){
    if(fillOn) meNClusters_onTrackRing_->Fill(onTrack); 
    if(fillOff) meNClusters_offTrackRing_->Fill(offTrack); 
  }
}
