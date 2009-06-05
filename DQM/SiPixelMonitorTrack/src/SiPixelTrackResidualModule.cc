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
// $Id: SiPixelTrackResidualModule.cc,v 1.1 2008/07/25 20:41:42 schuang Exp $


#include <string>
#include <iostream>

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
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


void SiPixelTrackResidualModule::book(const edm::ParameterSet& iConfig, int type) {
  DQMStore* dbe = edm::Service<DQMStore>().operator->();

  bool barrel = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  bool isHalfModule = false;
  if(barrel){
    isHalfModule = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).isHalfModule(); 
  }

  edm::InputTag src = iConfig.getParameter<edm::InputTag>("src");
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId(src.label());
  std::string hisID;

  if(type==0){
    hisID = theHistogramId->setHistoId("residualX",id_);
    meResidualX_ = dbe->book1D(hisID,"Hit-to-Track Residual in X",500,-5.,5.);
    meResidualX_->setAxisTitle("hit-to-track residual in x (cm)",1);
    
    hisID = theHistogramId->setHistoId("residualY",id_);
    meResidualY_ = dbe->book1D(hisID,"Hit-to-Track Residual in Y",500,-5.,5.);
    meResidualY_->setAxisTitle("hit-to-track residual in y (cm)",1);

    delete theHistogramId;
  }

  if(type==1 && barrel){
    uint32_t DBladder = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).ladderName();
    char sladder[80]; sprintf(sladder,"Ladder_%02i",DBladder);
    hisID = src.label() + "_" + sladder;
    if(isHalfModule) hisID += "H";
    else hisID += "F";

    meResidualXLad_ = dbe->book1D("residualX_"+hisID,"Hit-to-Track Residual in X",500,-5.,5.);
    meResidualXLad_->setAxisTitle("hit-to-track residual in x (cm)",1);
    
    meResidualYLad_ = dbe->book1D("residualY_"+hisID,"Hit-to-Track Residual in Y",500,-5.,5.);
    meResidualYLad_->setAxisTitle("hit-to-track residual in y (cm)",1);
  }

  if(type==2 && barrel){
    uint32_t DBlayer = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).layerName();
    char slayer[80]; sprintf(slayer,"Layer_%i",DBlayer);
    hisID = src.label() + "_" + slayer;

    meResidualXLay_ = dbe->book1D("residualX_"+hisID,"Hit-to-Track Residual in X",500,-5.,5.);
    meResidualXLay_->setAxisTitle("hit-to-track residual in x (cm)",1);
    
    meResidualYLay_ = dbe->book1D("residualY_"+hisID,"Hit-to-Track Residual in Y",500,-5.,5.);
    meResidualYLay_->setAxisTitle("hit-to-track residual in y (cm)",1);
  }

  if(type==3 && barrel){
    uint32_t DBmodule = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).moduleName();
    char smodule[80]; sprintf(smodule,"Ring_%i",DBmodule);
    hisID = src.label() + "_" + smodule;

    meResidualXPhi_ = dbe->book1D("residualX_"+hisID,"Hit-to-Track Residual in X",500,-5.,5.);
    meResidualXPhi_->setAxisTitle("hit-to-track residual in x (cm)",1);
    
    meResidualYPhi_ = dbe->book1D("residualY_"+hisID,"Hit-to-Track Residual in Y",500,-5.,5.);
    meResidualYPhi_->setAxisTitle("hit-to-track residual in y (cm)",1);
  }

  if(type==4 && endcap){
    uint32_t blade= PixelEndcapName::PixelEndcapName(DetId::DetId(id_)).bladeName();
    
    char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
    hisID = src.label() + "_" + sblade;

    meResidualXBlade_ = dbe->book1D("residualX_"+hisID,"Hit-to-Track Residual in X",500,-5.,5.);
    meResidualXBlade_->setAxisTitle("hit-to-track residual in x (cm)",1);

    meResidualYBlade_ = dbe->book1D("residualY_"+hisID,"Hit-to-Track Residual in Y",500,-5.,5.);
    meResidualYBlade_->setAxisTitle("hit-to-track residual in y (cm)",1);
  }

  if(type==5 && endcap){
    uint32_t disk = PixelEndcapName::PixelEndcapName(DetId::DetId(id_)).diskName();
    
    char sdisk[80]; sprintf(sdisk, "Disk_%i",disk);
    hisID = src.label() + "_" + sdisk;

    meResidualXDisk_ = dbe->book1D("residualX_"+hisID,"Hit-to-Track Residual in X",500,-5.,5.);
    meResidualXDisk_->setAxisTitle("hit-to-track residual in x (cm)",1);
    
    meResidualYDisk_ = dbe->book1D("residualY_"+hisID,"Hit-to-Track Residual in Y",500,-5.,5.);
    meResidualYDisk_->setAxisTitle("hit-to-track residual in y (cm)",1);
  }

  if(type==6 && endcap){
    uint32_t panel= PixelEndcapName::PixelEndcapName(DetId::DetId(id_)).pannelName();
    uint32_t module= PixelEndcapName::PixelEndcapName(DetId::DetId(id_)).plaquetteName();
    char slab[80]; sprintf(slab, "Panel_%i_Ring_%i",panel, module);
    hisID = src.label() + "_" + slab;

    meResidualXRing_ = dbe->book1D("residualX_"+hisID,"Hit-to-Track Residual in X",500,-5.,5.);
    meResidualXRing_->setAxisTitle("hit-to-track residual in x (cm)",1);
    
    meResidualYRing_ = dbe->book1D("residualY_"+hisID,"Hit-to-Track Residual in Y",500,-5.,5.);
    meResidualYRing_->setAxisTitle("hit-to-track residual in y (cm)",1);
  }

//   if(type==10){
//     dbe->setCurrentFolder("Pixel");
//     meNofTracks_ = dbe->book1D("ntracks_"+src.label(),"Number of Tracks",4,0,4);
//     meNofTracks_->setAxisTitle("number of tracks (all/pixel/bpix/fpix)",1);
//   }
  
}


void SiPixelTrackResidualModule::fill(const Measurement2DVector& residual, bool modon, bool ladon, bool layon, bool phion, bool bladeon, bool diskon, bool ringon) {

  bool barrel = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);

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


