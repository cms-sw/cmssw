#include "DQM/SiPixelMonitorRecHit/interface/SiPixelRecHitModule.h"
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
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
//
// Constructors
//
SiPixelRecHitModule::SiPixelRecHitModule() : id_(0) { }
///
SiPixelRecHitModule::SiPixelRecHitModule(const uint32_t& id) : 
  id_(id)
{ 
}

//
// Destructor
//
SiPixelRecHitModule::~SiPixelRecHitModule() {}
//
// Book histograms
//
void SiPixelRecHitModule::book(const edm::ParameterSet& iConfig, int type) {

  bool barrel = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  bool isHalfModule = false;
  if(barrel){
    isHalfModule = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).isHalfModule(); 
  }

  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = iConfig.getParameter<edm::InputTag>( "src" );
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );
  // Get DQM interface
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();


  if(type==0){
    // XYPosition
    hid = theHistogramId->setHistoId("xypos",id_);
    //std::cout << hid << " " << theHistogramId->getDataCollection(hid) << " " << theHistogramId->getRawId(hid) << std::endl;
    meXYPos_ = theDMBE->book2D(hid,"XY Position",100,-1.,1,100,-4,4);
    meXYPos_->setAxisTitle("X Position",1);
    meXYPos_->setAxisTitle("Y Position",2);
    hid = theHistogramId->setHistoId("ClustX",id_);
    meClustX_ = theDMBE->book1D(hid, "Cluster X size", 10, 0, 10);
    meClustX_->setAxisTitle("Cluster size X dimension", 1);
    hid = theHistogramId->setHistoId("ClustY",id_);
    meClustY_ = theDMBE->book1D(hid, "Cluster Y size", 25, 0., 25.);
    meClustY_->setAxisTitle("Cluster size Y dimension", 1); 
    hid = theHistogramId->setHistoId("nRecHits",id_);
    menRecHits_ = theDMBE->book1D(hid, "# of rechits in this module", 50, 0, 50);
    menRecHits_->setAxisTitle("number of rechits",1);  
    delete theHistogramId;
  }

  if(type==1 && barrel){
    uint32_t DBladder = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).ladderName();
    char sladder[80]; sprintf(sladder,"Ladder_%02i",DBladder);
    hid = src.label() + "_" + sladder;
    if(isHalfModule) hid += "H";
    else hid += "F";

    meXYPosLad_ = theDMBE->book2D("xypos_" + hid,"XY Position",100,-1.,1,100,-4,4);
    meXYPosLad_->setAxisTitle("X Position",1);
    meXYPosLad_->setAxisTitle("Y Position",2);

    meClustXLad_ = theDMBE->book1D("ClustX_" +hid, "Cluster X size", 10, 0, 10);
    meClustXLad_->setAxisTitle("Cluster size X dimension", 1);
    meClustYLad_ = theDMBE->book1D("ClustY_" +hid,"Cluster Y size", 25, 0.,25.);
    meClustYLad_->setAxisTitle("Cluster size Y dimension", 1);
    menRecHitsLad_ = theDMBE->book1D("nRecHits_"+hid, "# of rechits in this module", 50, 0, 50);
    menRecHitsLad_->setAxisTitle("number of rechits",1);

  }

  if(type==2 && barrel){
    
    uint32_t DBlayer = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).layerName();
    char slayer[80]; sprintf(slayer,"Layer_%i",DBlayer);
    hid = src.label() + "_" + slayer;
    
    meXYPosLay_ = theDMBE->book2D("xypos_" + hid,"XY Position",100,-1.,1,100,-4,4);
    meXYPosLay_->setAxisTitle("X Position",1);
    meXYPosLay_->setAxisTitle("Y Position",2);

    meClustXLay_ = theDMBE->book1D("ClustX_" +hid, "Cluster X size", 10, 0, 10);
    meClustXLay_->setAxisTitle("Cluster size X dimension", 1);
    meClustYLay_ = theDMBE->book1D("ClustY_" +hid,"Cluster Y size", 25, 0.,25.);
    meClustYLay_->setAxisTitle("Cluster size Y dimension", 1);
    menRecHitsLay_ = theDMBE->book1D("nRecHits_"+hid, "# of rechits in this module", 50, 0, 50);
    menRecHitsLay_->setAxisTitle("number of rechits",1);

  }

  if(type==3 && barrel){
    uint32_t DBmodule = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).moduleName();
    char smodule[80]; sprintf(smodule,"Ring_%i",DBmodule);
    hid = src.label() + "_" + smodule;

    meXYPosPhi_ = theDMBE->book2D("xypos_" + hid,"XY Position",100,-1.,1,100,-4,4);
    meXYPosPhi_->setAxisTitle("X Position",1);
    meXYPosPhi_->setAxisTitle("Y Position",2);

    meClustXPhi_ = theDMBE->book1D("ClustX_" +hid, "Cluster X size", 10, 0, 10);
    meClustXPhi_->setAxisTitle("Cluster size X dimension", 1);
    meClustYPhi_ = theDMBE->book1D("ClustY_" +hid,"Cluster Y size", 25, 0.,25.);
    meClustYPhi_->setAxisTitle("Cluster size Y dimension", 1);
    menRecHitsPhi_ = theDMBE->book1D("nRecHits_"+hid, "# of rechits in this module", 50, 0, 50);
    menRecHitsPhi_->setAxisTitle("number of rechits",1);

  }

  if(type==4 && endcap){
    uint32_t blade= PixelEndcapName::PixelEndcapName(DetId::DetId(id_)).bladeName();
    
    char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
    hid = src.label() + "_" + sblade;
//     meXYPosBlade_ = theDMBE->book2D("xypos_" + hid,"XY Position",100,-1.,1,100,-4,4);
//     meXYPosBlade_->setAxisTitle("X Position",1);
//     meXYPosBlade_->setAxisTitle("Y Position",2);

    meClustXBlade_ = theDMBE->book1D("ClustX_" +hid, "Cluster X size", 10, 0, 10);
    meClustXBlade_->setAxisTitle("Cluster size X dimension", 1);
    meClustYBlade_ = theDMBE->book1D("ClustY_" +hid,"Cluster Y size", 25, 0.,25.);
    meClustYBlade_->setAxisTitle("Cluster size Y dimension", 1);
    menRecHitsBlade_ = theDMBE->book1D("nRecHits_"+hid, "# of rechits in this module", 50, 0, 50);
    menRecHitsBlade_->setAxisTitle("number of rechits",1);

  }
  if(type==5 && endcap){
    uint32_t disk = PixelEndcapName::PixelEndcapName(DetId::DetId(id_)).diskName();
    
    char sdisk[80]; sprintf(sdisk, "Disk_%i",disk);
    hid = src.label() + "_" + sdisk;
//     meXYPosDisk_ = theDMBE->book2D("xypos_" + hid,"XY Position",100,-1.,1,100,-4,4);
//     meXYPosDisk_->setAxisTitle("X Position",1);
//     meXYPosDisk_->setAxisTitle("Y Position",2);

    meClustXDisk_ = theDMBE->book1D("ClustX_" +hid, "Cluster X size", 10, 0, 10);
    meClustXDisk_->setAxisTitle("Cluster size X dimension", 1);
    meClustYDisk_ = theDMBE->book1D("ClustY_" +hid,"Cluster Y size", 25, 0.,25.);
    meClustYDisk_->setAxisTitle("Cluster size Y dimension", 1);
    menRecHitsDisk_ = theDMBE->book1D("nRecHits_"+hid, "# of rechits in this module", 50, 0, 50);
    menRecHitsDisk_->setAxisTitle("number of rechits",1);

  }

  if(type==6 && endcap){
    uint32_t panel= PixelEndcapName::PixelEndcapName(DetId::DetId(id_)).pannelName();
    uint32_t module= PixelEndcapName::PixelEndcapName(DetId::DetId(id_)).plaquetteName();
    char slab[80]; sprintf(slab, "Panel_%i_Ring_%i",panel, module);
    hid = src.label() + "_" + slab;

    meXYPosRing_ = theDMBE->book2D("xypos_" + hid,"XY Position",100,-1.,1,100,-4,4);
    meXYPosRing_->setAxisTitle("X Position",1);
    meXYPosRing_->setAxisTitle("Y Position",2);

    meClustXRing_ = theDMBE->book1D("ClustX_" +hid, "Cluster X size", 10, 0, 10);
    meClustXRing_->setAxisTitle("Cluster size X dimension", 1);
    meClustYRing_ = theDMBE->book1D("ClustY_" +hid,"Cluster Y size", 25, 0.,25.);
    meClustYRing_->setAxisTitle("Cluster size Y dimension", 1);
    menRecHitsRing_ = theDMBE->book1D("nRecHits_"+hid, "# of rechits in this module", 50, 0, 50);
    menRecHitsRing_->setAxisTitle("number of rechits",1);

  }

}
//
// Fill histograms
//
void SiPixelRecHitModule::fill(const float& rechit_x, const float& rechit_y, const int& sizeX, const int& sizeY, bool modon, bool ladon, bool layon, bool phion, bool bladeon, bool diskon, bool ringon) {

   bool barrel = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
//   bool isHalfModule = false;
//   uint32_t DBladder = 0;
//   if(barrel){
//     isHalfModule = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).isHalfModule(); 
//     DBladder = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).ladderName();
//   }


/*
  edm::DetSetVector<PixelDigi>::const_iterator isearch = input.find(id_); // search  digis of detid
  
  if( isearch != input.end() ) {  // Not at empty iterator
    
    unsigned int numberOfDigis = 0;
    
    // Look at digis now
    edm::DetSet<PixelDigi>::const_iterator  di;
    for(di = isearch->data.begin(); di != isearch->data.end(); di++) {
      numberOfDigis++;
      int adc = di->adc();    // charge
      int col = di->column(); // column 
      int row = di->row();    // row
      (mePixDigis_)->Fill((float)col,(float)row);
      (meADC_)->Fill((float)adc);
    }
    (meNDigis_)->Fill((float)numberOfDigis);
    //std::cout<<"number of digis="<<numberOfDigis<<std::endl;
      
  }
  */
  //std::cout << rechit_x << " " << rechit_y << " " << sizeX << " " << sizeY << std::endl;
  if(modon){
    meXYPos_->Fill(rechit_x, rechit_y);
    meClustX_->Fill(sizeX);
    meClustY_->Fill(sizeY);
  }
  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;

  if(ladon && barrel){
    meXYPosLad_->Fill(rechit_x, rechit_y);
    meClustXLad_->Fill(sizeX);
    meClustYLad_->Fill(sizeY);
  }

  if(layon && barrel){
    meXYPosLay_->Fill(rechit_x, rechit_y);
    meClustXLay_->Fill(sizeX);
    meClustYLay_->Fill(sizeY);
  }

  if(phion && barrel){
    meXYPosPhi_->Fill(rechit_x, rechit_y);
    meClustXPhi_->Fill(sizeX);
    meClustYPhi_->Fill(sizeY);
  }

  if(bladeon && endcap){
    //meXYPosBlade_->Fill(rechit_x, rechit_y);
    meClustXBlade_->Fill(sizeX);
    meClustYBlade_->Fill(sizeY);
  }

  if(diskon && endcap){
    //meXYPosDisk_->Fill(rechit_x, rechit_y);
    meClustXDisk_->Fill(sizeX);
    meClustYDisk_->Fill(sizeY);
  }

  if(ringon && endcap){
    meXYPosRing_->Fill(rechit_x, rechit_y);
    meClustXRing_->Fill(sizeX);
    meClustYRing_->Fill(sizeY);
  }
}

void SiPixelRecHitModule::nfill(const int& nrec, bool modon, bool ladon, bool layon, bool phion, bool bladeon, bool diskon, bool ringon) {
  
  bool barrel = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);

  if(modon) menRecHits_->Fill(nrec);
  //barrel
  if(ladon && barrel) menRecHitsLad_->Fill(nrec);
  if(layon && barrel) menRecHitsLay_->Fill(nrec);
  if(phion && barrel) menRecHitsPhi_->Fill(nrec);
  //endcap
  if(bladeon && endcap) menRecHitsBlade_->Fill(nrec);
  if(diskon && endcap) menRecHitsDisk_->Fill(nrec);
  if(ringon && endcap) menRecHitsRing_->Fill(nrec);
}
