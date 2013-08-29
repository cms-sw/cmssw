// Package:    SiPixelMonitorTrack
// Class:      SiPixelHitEfficiencyModule
// 
// class SiPixelHitEfficiencyModule SiPixelHitEfficiencyModule.cc 
//       DQM/SiPixelMonitorTrack/src/SiPixelHitEfficiencyModule.cc
//
// Description: SiPixel hit efficiency data quality monitoring modules
// Implementation: prototype -> improved -> never final - end of the 1st step 
//
// Original Authors: Romain Rougny & Luca Mucibello
//         Created: Mar Nov 10 13:29:00 CET nbinangle9



#include <string>
#include <iostream>

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
#include "DQM/SiPixelMonitorTrack/interface/SiPixelHitEfficiencyModule.h"

// Data Formats
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


using namespace std; 


SiPixelHitEfficiencyModule::SiPixelHitEfficiencyModule() : id_(0) {
  bBookTracks = true;
}


SiPixelHitEfficiencyModule::SiPixelHitEfficiencyModule(uint32_t id) : id_(id) { 
  bBookTracks = true;
}


SiPixelHitEfficiencyModule::~SiPixelHitEfficiencyModule() { 
 
}


void SiPixelHitEfficiencyModule::book(const edm::ParameterSet& iConfig, int type, bool isUpgrade) {
  DQMStore* dbe = edm::Service<DQMStore>().operator->();

  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  bool isHalfModule = false;
  if(barrel){
    if (!isUpgrade) {
    isHalfModule = PixelBarrelName(DetId(id_)).isHalfModule(); 
    } else if (isUpgrade) {
      isHalfModule = PixelBarrelNameUpgrade(DetId(id_)).isHalfModule(); 
    }
  }

  edm::InputTag src = iConfig.getParameter<edm::InputTag>("src");
  debug_ = iConfig.getUntrackedParameter<bool>("debug",false);
  updateEfficiencies = iConfig.getUntrackedParameter<bool>("updateEfficiencies",false);
  std::string hisID;
  
  int nbinangle = 28;
  int nbinX = 45;
  int nbinY = 40;

  if(type==0){
    SiPixelHistogramId* theHistogramId = new SiPixelHistogramId(src.label());
    
    if(updateEfficiencies){
      //EFFICIENCY
      hisID = theHistogramId->setHistoId("efficiency",id_);
      meEfficiency_ = dbe->book1D(hisID,"Hit efficiency",1,0,1.);
      meEfficiency_->setAxisTitle("Hit efficiency",1);
    
      hisID = theHistogramId->setHistoId("efficiencyX",id_);
      meEfficiencyX_ = dbe->book1D(hisID,"Hit efficiency in X",nbinX,-1.5,1.5);
      meEfficiencyX_->setAxisTitle("Hit efficiency in X",1);
    
      hisID = theHistogramId->setHistoId("efficiencyY",id_);
      meEfficiencyY_ = dbe->book1D(hisID,"Hit efficiency in Y",nbinY,-4.,4.);
      meEfficiencyY_->setAxisTitle("Hit efficiency in Y",1);
    
      hisID = theHistogramId->setHistoId("efficiencyAlpha",id_);
      meEfficiencyAlpha_ = dbe->book1D(hisID,"Hit efficiency in Alpha",nbinangle,-3.5,3.5);
      meEfficiencyAlpha_->setAxisTitle("Hit efficiency in Alpha",1);
    
      hisID = theHistogramId->setHistoId("efficiencyBeta",id_);
      meEfficiencyBeta_ = dbe->book1D(hisID,"Hit efficiency in Beta",nbinangle,-3.5,3.5);
      meEfficiencyBeta_->setAxisTitle("Hit efficiency in Beta",1);
    }

    //VALID
    hisID = theHistogramId->setHistoId("valid",id_);
    meValid_ = dbe->book1D(hisID,"# Valid hits",1,0,1.);
    meValid_->setAxisTitle("# Valid hits",1);
    
    /*hisID = theHistogramId->setHistoId("validX",id_);
    meValidX_ = dbe->book1D(hisID,"# Valid hits in X",nbinX,-1.5,1.5);
    meValidX_->setAxisTitle("# Valid hits in X",1);
    
    hisID = theHistogramId->setHistoId("validY",id_);
    meValidY_ = dbe->book1D(hisID,"# Valid hits in Y",nbinY,-4.,4.);
    meValidY_->setAxisTitle("# Valid hits in Y",1);
    
    hisID = theHistogramId->setHistoId("validAlpha",id_);
    meValidAlpha_ = dbe->book1D(hisID,"# Valid hits in Alpha",nbinangle,-3.5,3.5);
    meValidAlpha_->setAxisTitle("# Valid hits in Alpha",1);
    
    hisID = theHistogramId->setHistoId("validBeta",id_);
    meValidBeta_ = dbe->book1D(hisID,"# Valid hits in Beta",nbinangle,-3.5,3.5);
    meValidBeta_->setAxisTitle("# Valid hits in Beta",1);*/

    //MISSING
    hisID = theHistogramId->setHistoId("missing",id_);
    meMissing_ = dbe->book1D(hisID,"# Missing hits",1,0,1.);
    meMissing_->setAxisTitle("# Missing hits",1);
    
    /*hisID = theHistogramId->setHistoId("missingX",id_);
    meMissingX_ = dbe->book1D(hisID,"# Missing hits in X",nbinX,-1.5,1.5);
    meMissingX_->setAxisTitle("# Missing hits in X",1);
    
    hisID = theHistogramId->setHistoId("missingY",id_);
    meMissingY_ = dbe->book1D(hisID,"# Missing hits in Y",nbinY,-4.,4.);
    meMissingY_->setAxisTitle("# Missing hits in Y",1);
    
    hisID = theHistogramId->setHistoId("missingAlpha",id_);
    meMissingAlpha_ = dbe->book1D(hisID,"# Missing hits in Alpha",nbinangle,-3.5,3.5);
    meMissingAlpha_->setAxisTitle("# Missing hits in Alpha",1);
    
    hisID = theHistogramId->setHistoId("missingBeta",id_);
    meMissingBeta_ = dbe->book1D(hisID,"# Missing hits in Beta",nbinangle,-3.5,3.5);
    meMissingBeta_->setAxisTitle("# Missing hits in Beta",1);*/
    
    delete theHistogramId;
  }

  if(type==1 && barrel){
    uint32_t DBladder;
    if (!isUpgrade) { DBladder = PixelBarrelName(DetId(id_)).ladderName(); }
    else if (isUpgrade) { DBladder = PixelBarrelNameUpgrade(DetId(id_)).ladderName(); }
    char sladder[80]; sprintf(sladder,"Ladder_%02i",DBladder);
    hisID = src.label() + "_" + sladder;
    if(isHalfModule) hisID += "H";
    else hisID += "F";
    
    if(updateEfficiencies){
      //EFFICIENCY
      meEfficiencyLad_ = dbe->book1D("efficiency_"+hisID,"Hit efficiency",1,0,1.);
      meEfficiencyLad_->setAxisTitle("Hit efficiency",1);
    
      meEfficiencyXLad_ = dbe->book1D("efficiencyX_"+hisID,"Hit efficiency in X",nbinX,-1.5,1.5);
      meEfficiencyXLad_->setAxisTitle("Hit efficiency in X",1);
    
      meEfficiencyYLad_ = dbe->book1D("efficiencyY_"+hisID,"Hit efficiency in Y",nbinY,-4.,4.);
      meEfficiencyYLad_->setAxisTitle("Hit efficiency in Y",1);
    
      meEfficiencyAlphaLad_ = dbe->book1D("efficiencyAlpha_"+hisID,"Hit efficiency in Alpha",nbinangle,-3.5,3.5);
      meEfficiencyAlphaLad_->setAxisTitle("Hit efficiency in Alpha",1);
    
      meEfficiencyBetaLad_ = dbe->book1D("efficiencyBeta_"+hisID,"Hit efficiency in Beta",nbinangle,-3.5,3.5);
      meEfficiencyBetaLad_->setAxisTitle("Hit efficiency in Beta",1);
    }
    
    //VALID
    meValidLad_ = dbe->book1D("valid_"+hisID,"# Valid hits",1,0,1.);
    meValidLad_->setAxisTitle("# Valid hits",1);
    
    meValidXLad_ = dbe->book1D("validX_"+hisID,"# Valid hits in X",nbinX,-1.5,1.5);
    meValidXLad_->setAxisTitle("# Valid hits in X",1);
    
    meValidYLad_ = dbe->book1D("validY_"+hisID,"# Valid hits in Y",nbinY,-4.,4.);
    meValidYLad_->setAxisTitle("# Valid hits in Y",1);
    
    meValidAlphaLad_ = dbe->book1D("validAlpha_"+hisID,"# Valid hits in Alpha",nbinangle,-3.5,3.5);
    meValidAlphaLad_->setAxisTitle("# Valid hits in Alpha",1);
    
    meValidBetaLad_ = dbe->book1D("validBeta_"+hisID,"# Valid hits in Beta",nbinangle,-3.5,3.5);
    meValidBetaLad_->setAxisTitle("# Valid hits in Beta",1);

    //MISSING
    meMissingLad_ = dbe->book1D("missing_"+hisID,"# Missing hits",1,0,1.);
    meMissingLad_->setAxisTitle("# Missing hits",1);
    
    meMissingXLad_ = dbe->book1D("missingX_"+hisID,"# Missing hits in X",nbinX,-1.5,1.5);
    meMissingXLad_->setAxisTitle("# Missing hits in X",1);
    
    meMissingYLad_ = dbe->book1D("missingY_"+hisID,"# Missing hits in Y",nbinY,-4.,4.);
    meMissingYLad_->setAxisTitle("# Missing hits in Y",1);
    
    meMissingAlphaLad_ = dbe->book1D("missingAlpha_"+hisID,"# Missing hits in Alpha",nbinangle,-3.5,3.5);
    meMissingAlphaLad_->setAxisTitle("# Missing hits in Alpha",1);
    
    meMissingBetaLad_ = dbe->book1D("missingBeta_"+hisID,"# Missing hits in Beta",nbinangle,-3.5,3.5);
    meMissingBetaLad_->setAxisTitle("# Missing hits in Beta",1);
  }
  
  if(type==2 && barrel){
    uint32_t DBlayer;
    if (!isUpgrade) { DBlayer = PixelBarrelName(DetId(id_)).layerName(); }
    else if (isUpgrade) { DBlayer = PixelBarrelNameUpgrade(DetId(id_)).layerName(); }
    char slayer[80]; sprintf(slayer,"Layer_%i",DBlayer);
    hisID = src.label() + "_" + slayer;


    if(updateEfficiencies){
      //EFFICIENCY
      meEfficiencyLay_ = dbe->book1D("efficiency_"+hisID,"Hit efficiency",1,0,1.);
      meEfficiencyLay_->setAxisTitle("Hit efficiency",1);
    
      meEfficiencyXLay_ = dbe->book1D("efficiencyX_"+hisID,"Hit efficiency in X",nbinX,-1.5,1.5);
      meEfficiencyXLay_->setAxisTitle("Hit efficiency in X",1);
    
      meEfficiencyYLay_ = dbe->book1D("efficiencyY_"+hisID,"Hit efficiency in Y",nbinY,-4.,4.);
      meEfficiencyYLay_->setAxisTitle("Hit efficiency in Y",1);
    
      meEfficiencyAlphaLay_ = dbe->book1D("efficiencyAlpha_"+hisID,"Hit efficiency in Alpha",nbinangle,-3.5,3.5);
      meEfficiencyAlphaLay_->setAxisTitle("Hit efficiency in Alpha",1);
    
      meEfficiencyBetaLay_ = dbe->book1D("efficiencyBeta_"+hisID,"Hit efficiency in Beta",nbinangle,-3.5,3.5);
      meEfficiencyBetaLay_->setAxisTitle("Hit efficiency in Beta",1);
    }
    
    //VALID
    meValidLay_ = dbe->book1D("valid_"+hisID,"# Valid hits",1,0,1.);
    meValidLay_->setAxisTitle("# Valid hits",1);
    
    meValidXLay_ = dbe->book1D("validX_"+hisID,"# Valid hits in X",nbinX,-1.5,1.5);
    meValidXLay_->setAxisTitle("# Valid hits in X",1);
    
    meValidYLay_ = dbe->book1D("validY_"+hisID,"# Valid hits in Y",nbinY,-4.,4.);
    meValidYLay_->setAxisTitle("# Valid hits in Y",1);
    
    meValidAlphaLay_ = dbe->book1D("validAlpha_"+hisID,"# Valid hits in Alpha",nbinangle,-3.5,3.5);
    meValidAlphaLay_->setAxisTitle("# Valid hits in Alpha",1);
    
    meValidBetaLay_ = dbe->book1D("validBeta_"+hisID,"# Valid hits in Beta",nbinangle,-3.5,3.5);
    meValidBetaLay_->setAxisTitle("# Valid hits in Beta",1);

    //MISSING
    meMissingLay_ = dbe->book1D("missing_"+hisID,"# Missing hits",1,0,1.);
    meMissingLay_->setAxisTitle("# Missing hits",1);
    
    meMissingXLay_ = dbe->book1D("missingX_"+hisID,"# Missing hits in X",nbinX,-1.5,1.5);
    meMissingXLay_->setAxisTitle("# Missing hits in X",1);
    
    meMissingYLay_ = dbe->book1D("missingY_"+hisID,"# Missing hits in Y",nbinY,-4.,4.);
    meMissingYLay_->setAxisTitle("# Missing hits in Y",1);
    
    meMissingAlphaLay_ = dbe->book1D("missingAlpha_"+hisID,"# Missing hits in Alpha",nbinangle,-3.5,3.5);
    meMissingAlphaLay_->setAxisTitle("# Missing hits in Alpha",1);
    
    meMissingBetaLay_ = dbe->book1D("missingBeta_"+hisID,"# Missing hits in Beta",nbinangle,-3.5,3.5);
    meMissingBetaLay_->setAxisTitle("# Missing hits in Beta",1);
  }
  
  if(type==3 && barrel){
    uint32_t DBmodule;
    if (!isUpgrade) { DBmodule = PixelBarrelName(DetId(id_)).moduleName(); }
    else if (isUpgrade) { DBmodule = PixelBarrelNameUpgrade(DetId(id_)).moduleName(); }
    char smodule[80]; sprintf(smodule,"Ring_%i",DBmodule);
    hisID = src.label() + "_" + smodule;
    
    if(updateEfficiencies){
      //EFFICIENCY
      meEfficiencyPhi_ = dbe->book1D("efficiency_"+hisID,"Hit efficiency",1,0,1.);
      meEfficiencyPhi_->setAxisTitle("Hit efficiency",1);
    
      meEfficiencyXPhi_ = dbe->book1D("efficiencyX_"+hisID,"Hit efficiency in X",nbinX,-1.5,1.5);
      meEfficiencyXPhi_->setAxisTitle("Hit efficiency in X",1);
    
      meEfficiencyYPhi_ = dbe->book1D("efficiencyY_"+hisID,"Hit efficiency in Y",nbinY,-4.,4.);
      meEfficiencyYPhi_->setAxisTitle("Hit efficiency in Y",1);
    
      meEfficiencyAlphaPhi_ = dbe->book1D("efficiencyAlpha_"+hisID,"Hit efficiency in Alpha",nbinangle,-3.5,3.5);
      meEfficiencyAlphaPhi_->setAxisTitle("Hit efficiency in Alpha",1);
    
      meEfficiencyBetaPhi_ = dbe->book1D("efficiencyBeta_"+hisID,"Hit efficiency in Beta",nbinangle,-3.5,3.5);
      meEfficiencyBetaPhi_->setAxisTitle("Hit efficiency in Beta",1);
    }
    
    //VALID
    meValidPhi_ = dbe->book1D("valid_"+hisID,"# Valid hits",1,0,1.);
    meValidPhi_->setAxisTitle("# Valid hits",1);
    
    meValidXPhi_ = dbe->book1D("validX_"+hisID,"# Valid hits in X",nbinX,-1.5,1.5);
    meValidXPhi_->setAxisTitle("# Valid hits in X",1);
    
    meValidYPhi_ = dbe->book1D("validY_"+hisID,"# Valid hits in Y",nbinY,-4.,4.);
    meValidYPhi_->setAxisTitle("# Valid hits in Y",1);
    
    meValidAlphaPhi_ = dbe->book1D("validAlpha_"+hisID,"# Valid hits in Alpha",nbinangle,-3.5,3.5);
    meValidAlphaPhi_->setAxisTitle("# Valid hits in Alpha",1);
    
    meValidBetaPhi_ = dbe->book1D("validBeta_"+hisID,"# Valid hits in Beta",nbinangle,-3.5,3.5);
    meValidBetaPhi_->setAxisTitle("# Valid hits in Beta",1);

    //MISSING
    meMissingPhi_ = dbe->book1D("missing_"+hisID,"# Missing hits",1,0,1.);
    meMissingPhi_->setAxisTitle("# Missing hits",1);
    
    meMissingXPhi_ = dbe->book1D("missingX_"+hisID,"# Missing hits in X",nbinX,-1.5,1.5);
    meMissingXPhi_->setAxisTitle("# Missing hits in X",1);
    
    meMissingYPhi_ = dbe->book1D("missingY_"+hisID,"# Missing hits in Y",nbinY,-4.,4.);
    meMissingYPhi_->setAxisTitle("# Missing hits in Y",1);
    
    meMissingAlphaPhi_ = dbe->book1D("missingAlpha_"+hisID,"# Missing hits in Alpha",nbinangle,-3.5,3.5);
    meMissingAlphaPhi_->setAxisTitle("# Missing hits in Alpha",1);
    
    meMissingBetaPhi_ = dbe->book1D("missingBeta_"+hisID,"# Missing hits in Beta",nbinangle,-3.5,3.5);
    meMissingBetaPhi_->setAxisTitle("# Missing hits in Beta",1); 
  }
  
  if(type==4 && endcap){
    uint32_t blade;
    if (!isUpgrade) { blade= PixelEndcapName(DetId(id_)).bladeName(); }
    else if (isUpgrade) { blade= PixelEndcapNameUpgrade(DetId(id_)).bladeName(); }
    
    char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
    hisID = src.label() + "_" + sblade;
    
    if(updateEfficiencies){
      //EFFICIENCY
      meEfficiencyBlade_ = dbe->book1D("efficiency_"+hisID,"Hit efficiency",1,0,1.);
      meEfficiencyBlade_->setAxisTitle("Hit efficiency",1);
    
      meEfficiencyXBlade_ = dbe->book1D("efficiencyX_"+hisID,"Hit efficiency in X",nbinX,-1.5,1.5);
      meEfficiencyXBlade_->setAxisTitle("Hit efficiency in X",1);
    
      meEfficiencyYBlade_ = dbe->book1D("efficiencyY_"+hisID,"Hit efficiency in Y",nbinY,-4.,4.);
      meEfficiencyYBlade_->setAxisTitle("Hit efficiency in Y",1);
    
      meEfficiencyAlphaBlade_ = dbe->book1D("efficiencyAlpha_"+hisID,"Hit efficiency in Alpha",nbinangle,-3.5,3.5);
      meEfficiencyAlphaBlade_->setAxisTitle("Hit efficiency in Alpha",1);
    
      meEfficiencyBetaBlade_ = dbe->book1D("efficiencyBeta_"+hisID,"Hit efficiency in Beta",nbinangle,-3.5,3.5);
      meEfficiencyBetaBlade_->setAxisTitle("Hit efficiency in Beta",1);
    }
    
    //VALID
    meValidBlade_ = dbe->book1D("valid_"+hisID,"# Valid hits",1,0,1.);
    meValidBlade_->setAxisTitle("# Valid hits",1);
    
    meValidXBlade_ = dbe->book1D("validX_"+hisID,"# Valid hits in X",nbinX,-1.5,1.5);
    meValidXBlade_->setAxisTitle("# Valid hits in X",1);
    
    meValidYBlade_ = dbe->book1D("validY_"+hisID,"# Valid hits in Y",nbinY,-4.,4.);
    meValidYBlade_->setAxisTitle("# Valid hits in Y",1);
    
    meValidAlphaBlade_ = dbe->book1D("validAlpha_"+hisID,"# Valid hits in Alpha",nbinangle,-3.5,3.5);
    meValidAlphaBlade_->setAxisTitle("# Valid hits in Alpha",1);
    
    meValidBetaBlade_ = dbe->book1D("validBeta_"+hisID,"# Valid hits in Beta",nbinangle,-3.5,3.5);
    meValidBetaBlade_->setAxisTitle("# Valid hits in Beta",1);

    //MISSING
    meMissingBlade_ = dbe->book1D("missing_"+hisID,"# Missing hits",1,0,1.);
    meMissingBlade_->setAxisTitle("# Missing hits",1);
    
    meMissingXBlade_ = dbe->book1D("missingX_"+hisID,"# Missing hits in X",nbinX,-1.5,1.5);
    meMissingXBlade_->setAxisTitle("# Missing hits in X",1);
    
    meMissingYBlade_ = dbe->book1D("missingY_"+hisID,"# Missing hits in Y",nbinY,-4.,4.);
    meMissingYBlade_->setAxisTitle("# Missing hits in Y",1);
    
    meMissingAlphaBlade_ = dbe->book1D("missingAlpha_"+hisID,"# Missing hits in Alpha",nbinangle,-3.5,3.5);
    meMissingAlphaBlade_->setAxisTitle("# Missing hits in Alpha",1);
    
    meMissingBetaBlade_ = dbe->book1D("missingBeta_"+hisID,"# Missing hits in Beta",nbinangle,-3.5,3.5);
    meMissingBetaBlade_->setAxisTitle("# Missing hits in Beta",1); 
  }
  
  if(type==5 && endcap){
    uint32_t disk;
    if (!isUpgrade) { disk = PixelEndcapName(DetId(id_)).diskName(); }
    else if (isUpgrade) { disk = PixelEndcapNameUpgrade(DetId(id_)).diskName(); }
    
    char sdisk[80]; sprintf(sdisk, "Disk_%i",disk);
    hisID = src.label() + "_" + sdisk;
    
    if(updateEfficiencies){
      //EFFICIENCY
      meEfficiencyDisk_ = dbe->book1D("efficiency_"+hisID,"Hit efficiency",1,0,1.);
      meEfficiencyDisk_->setAxisTitle("Hit efficiency",1);
    
      meEfficiencyXDisk_ = dbe->book1D("efficiencyX_"+hisID,"Hit efficiency in X",nbinX,-1.5,1.5);
      meEfficiencyXDisk_->setAxisTitle("Hit efficiency in X",1);
    
      meEfficiencyYDisk_ = dbe->book1D("efficiencyY_"+hisID,"Hit efficiency in Y",nbinY,-4.,4.);
      meEfficiencyYDisk_->setAxisTitle("Hit efficiency in Y",1);
    
      meEfficiencyAlphaDisk_ = dbe->book1D("efficiencyAlpha_"+hisID,"Hit efficiency in Alpha",nbinangle,-3.5,3.5);
      meEfficiencyAlphaDisk_->setAxisTitle("Hit efficiency in Alpha",1);
    
      meEfficiencyBetaDisk_ = dbe->book1D("efficiencyBeta_"+hisID,"Hit efficiency in Beta",nbinangle,-3.5,3.5);
      meEfficiencyBetaDisk_->setAxisTitle("Hit efficiency in Beta",1);
    }
    
    //VALID
    meValidDisk_ = dbe->book1D("valid_"+hisID,"# Valid hits",1,0,1.);
    meValidDisk_->setAxisTitle("# Valid hits",1);
    
    meValidXDisk_ = dbe->book1D("validX_"+hisID,"# Valid hits in X",nbinX,-1.5,1.5);
    meValidXDisk_->setAxisTitle("# Valid hits in X",1);
    
    meValidYDisk_ = dbe->book1D("validY_"+hisID,"# Valid hits in Y",nbinY,-4.,4.);
    meValidYDisk_->setAxisTitle("# Valid hits in Y",1);
    
    meValidAlphaDisk_ = dbe->book1D("validAlpha_"+hisID,"# Valid hits in Alpha",nbinangle,-3.5,3.5);
    meValidAlphaDisk_->setAxisTitle("# Valid hits in Alpha",1);
    
    meValidBetaDisk_ = dbe->book1D("validBeta_"+hisID,"# Valid hits in Beta",nbinangle,-3.5,3.5);
    meValidBetaDisk_->setAxisTitle("# Valid hits in Beta",1);

    //MISSING
    meMissingDisk_ = dbe->book1D("missing_"+hisID,"# Missing hits",1,0,1.);
    meMissingDisk_->setAxisTitle("# Missing hits",1);
    
    meMissingXDisk_ = dbe->book1D("missingX_"+hisID,"# Missing hits in X",nbinX,-1.5,1.5);
    meMissingXDisk_->setAxisTitle("# Missing hits in X",1);
    
    meMissingYDisk_ = dbe->book1D("missingY_"+hisID,"# Missing hits in Y",nbinY,-4.,4.);
    meMissingYDisk_->setAxisTitle("# Missing hits in Y",1);
    
    meMissingAlphaDisk_ = dbe->book1D("missingAlpha_"+hisID,"# Missing hits in Alpha",nbinangle,-3.5,3.5);
    meMissingAlphaDisk_->setAxisTitle("# Missing hits in Alpha",1);
    
    meMissingBetaDisk_ = dbe->book1D("missingBeta_"+hisID,"# Missing hits in Beta",nbinangle,-3.5,3.5);
    meMissingBetaDisk_->setAxisTitle("# Missing hits in Beta",1);
  }
    
   
  if(type==6 && endcap){
    uint32_t panel;
    uint32_t module;
    if (!isUpgrade) {
      panel= PixelEndcapName(DetId(id_)).pannelName();
      module= PixelEndcapName(DetId(id_)).plaquetteName();
    } else if (isUpgrade) {
      panel= PixelEndcapNameUpgrade(DetId(id_)).pannelName();
      module= PixelEndcapNameUpgrade(DetId(id_)).plaquetteName();
    }
    
    char slab[80]; sprintf(slab, "Panel_%i_Ring_%i",panel, module);
    hisID = src.label() + "_" + slab;
    
    if(updateEfficiencies){
      //EFFICIENCY
      meEfficiencyRing_ = dbe->book1D("efficiency_"+hisID,"Hit efficiency",1,0,1.);
      meEfficiencyRing_->setAxisTitle("Hit efficiency",1);
    
      meEfficiencyXRing_ = dbe->book1D("efficiencyX_"+hisID,"Hit efficiency in X",nbinX,-1.5,1.5);
      meEfficiencyXRing_->setAxisTitle("Hit efficiency in X",1);
    
      meEfficiencyYRing_ = dbe->book1D("efficiencyY_"+hisID,"Hit efficiency in Y",nbinY,-4.,4.);
      meEfficiencyYRing_->setAxisTitle("Hit efficiency in Y",1);
    
      meEfficiencyAlphaRing_ = dbe->book1D("efficiencyAlpha_"+hisID,"Hit efficiency in Alpha",nbinangle,-3.5,3.5);
      meEfficiencyAlphaRing_->setAxisTitle("Hit efficiency in Alpha",1);
    
      meEfficiencyBetaRing_ = dbe->book1D("efficiencyBeta_"+hisID,"Hit efficiency in Beta",nbinangle,-3.5,3.5);
      meEfficiencyBetaRing_->setAxisTitle("Hit efficiency in Beta",1);
    }
    
    //VALID
    meValidRing_ = dbe->book1D("valid_"+hisID,"# Valid hits",1,0,1.);
    meValidRing_->setAxisTitle("# Valid hits",1);
    
    meValidXRing_ = dbe->book1D("validX_"+hisID,"# Valid hits in X",nbinX,-1.5,1.5);
    meValidXRing_->setAxisTitle("# Valid hits in X",1);
    
    meValidYRing_ = dbe->book1D("validY_"+hisID,"# Valid hits in Y",nbinY,-4.,4.);
    meValidYRing_->setAxisTitle("# Valid hits in Y",1);
    
    meValidAlphaRing_ = dbe->book1D("validAlpha_"+hisID,"# Valid hits in Alpha",nbinangle,-3.5,3.5);
    meValidAlphaRing_->setAxisTitle("# Valid hits in Alpha",1);
    
    meValidBetaRing_ = dbe->book1D("validBeta_"+hisID,"# Valid hits in Beta",nbinangle,-3.5,3.5);
    meValidBetaRing_->setAxisTitle("# Valid hits in Beta",1);

    //MISSING
    meMissingRing_ = dbe->book1D("missing_"+hisID,"# Missing hits",1,0,1.);
    meMissingRing_->setAxisTitle("# Missing hits",1);
    
    meMissingXRing_ = dbe->book1D("missingX_"+hisID,"# Missing hits in X",nbinX,-1.5,1.5);
    meMissingXRing_->setAxisTitle("# Missing hits in X",1);
    
    meMissingYRing_ = dbe->book1D("missingY_"+hisID,"# Missing hits in Y",nbinY,-4.,4.);
    meMissingYRing_->setAxisTitle("# Missing hits in Y",1);
    
    meMissingAlphaRing_ = dbe->book1D("missingAlpha_"+hisID,"# Missing hits in Alpha",nbinangle,-3.5,3.5);
    meMissingAlphaRing_->setAxisTitle("# Missing hits in Alpha",1);
    
    meMissingBetaRing_ = dbe->book1D("missingBeta_"+hisID,"# Missing hits in Beta",nbinangle,-3.5,3.5);
    meMissingBetaRing_->setAxisTitle("# Missing hits in Beta",1);
  }
}


void SiPixelHitEfficiencyModule::fill(LocalTrajectoryParameters ltp, bool isHitValid, bool modon, bool ladon, bool layon, bool phion, bool bladeon, bool diskon, bool ringon) {  
  
  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);

  LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
  float prediction_alpha = atan2(localDir.z(), localDir.x());
  float prediction_beta = atan2(localDir.z(), localDir.y());
  float prediction_x = ltp.position().x();
  float prediction_y = ltp.position().y();
  
  if(isHitValid){
    if(modon){
      meValid_->Fill(0.5);
      /*meValidX_->Fill(prediction_x);
      meValidY_->Fill(prediction_y);
      meValidAlpha_->Fill(prediction_alpha);
      meValidBeta_->Fill(prediction_beta);*/
    }
    if(barrel && ladon){
      meValidLad_->Fill(0.5);
      meValidXLad_->Fill(prediction_x);
      meValidYLad_->Fill(prediction_y);
      meValidAlphaLad_->Fill(prediction_alpha);
      meValidBetaLad_->Fill(prediction_beta);
    }
    if(barrel && layon){
      meValidLay_->Fill(0.5);
      meValidXLay_->Fill(prediction_x);
      meValidYLay_->Fill(prediction_y);
      meValidAlphaLay_->Fill(prediction_alpha);
      meValidBetaLay_->Fill(prediction_beta);
    }   
    if(barrel && phion){
      meValidPhi_->Fill(0.5);
      meValidXPhi_->Fill(prediction_x);
      meValidYPhi_->Fill(prediction_y);
      meValidAlphaPhi_->Fill(prediction_alpha);
      meValidBetaPhi_->Fill(prediction_beta);
    }
    if(endcap && bladeon){
      meValidBlade_->Fill(0.5);
      meValidXBlade_->Fill(prediction_x);
      meValidYBlade_->Fill(prediction_y);
      meValidAlphaBlade_->Fill(prediction_alpha);
      meValidBetaBlade_->Fill(prediction_beta);
    }
    if(endcap && diskon){
      meValidDisk_->Fill(0.5);
      meValidXDisk_->Fill(prediction_x);
      meValidYDisk_->Fill(prediction_y);
      meValidAlphaDisk_->Fill(prediction_alpha);
      meValidBetaDisk_->Fill(prediction_beta);
    }
    if(endcap && ringon){
      meValidRing_->Fill(0.5);
      meValidXRing_->Fill(prediction_x);
      meValidYRing_->Fill(prediction_y);
      meValidAlphaRing_->Fill(prediction_alpha);
      meValidBetaRing_->Fill(prediction_beta);
    }
  }
  else {
    if(modon){
      meMissing_->Fill(0.5);
      /*meMissingX_->Fill(prediction_x);
      meMissingY_->Fill(prediction_y);
      meMissingAlpha_->Fill(prediction_alpha);
      meMissingBeta_->Fill(prediction_beta);*/
    }
    if(barrel && ladon){
      meMissingLad_->Fill(0.5);
      meMissingXLad_->Fill(prediction_x);
      meMissingYLad_->Fill(prediction_y);
      meMissingAlphaLad_->Fill(prediction_alpha);
      meMissingBetaLad_->Fill(prediction_beta);
    }
    if(barrel && layon){
      meMissingLay_->Fill(0.5);
      meMissingXLay_->Fill(prediction_x);
      meMissingYLay_->Fill(prediction_y);
      meMissingAlphaLay_->Fill(prediction_alpha);
      meMissingBetaLay_->Fill(prediction_beta);
    }   
    if(barrel && phion){
      meMissingPhi_->Fill(0.5);
      meMissingXPhi_->Fill(prediction_x);
      meMissingYPhi_->Fill(prediction_y);
      meMissingAlphaPhi_->Fill(prediction_alpha);
      meMissingBetaPhi_->Fill(prediction_beta);
    }
    if(endcap && bladeon){
      meMissingBlade_->Fill(0.5);
      meMissingXBlade_->Fill(prediction_x);
      meMissingYBlade_->Fill(prediction_y);
      meMissingAlphaBlade_->Fill(prediction_alpha);
      meMissingBetaBlade_->Fill(prediction_beta);
    }
    if(endcap && diskon){
      meMissingDisk_->Fill(0.5);
      meMissingXDisk_->Fill(prediction_x);
      meMissingYDisk_->Fill(prediction_y);
      meMissingAlphaDisk_->Fill(prediction_alpha);
      meMissingBetaDisk_->Fill(prediction_beta);
    }
    if(endcap && ringon){
      meMissingRing_->Fill(0.5);
      meMissingXRing_->Fill(prediction_x);
      meMissingYRing_->Fill(prediction_y);
      meMissingAlphaRing_->Fill(prediction_alpha);
      meMissingBetaRing_->Fill(prediction_beta);
    }
  }
  
  if(updateEfficiencies)
    computeEfficiencies(modon, ladon, layon, phion, bladeon, diskon, ringon);
}

void SiPixelHitEfficiencyModule::computeEfficiencies(bool modon, bool ladon, bool layon, bool phion, bool bladeon, bool diskon, bool ringon) {
  
  if(debug_)
    std::cout<<"Now Filling histos for detid "<<id_<<std::endl;
  
  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  
  if(modon){
    meEfficiency_->setBinContent(1,(eff(meValid_->getBinContent(1),meMissing_->getBinContent(1))).first);
    meEfficiency_->setBinError(1,(eff(meValid_->getBinContent(1),meMissing_->getBinContent(1))).second);
    /*for(int i=1;i<=meValidX_->getNbinsX();++i){
      meEfficiencyX_->setBinContent(i,(eff(meValidX_->getBinContent(i),meMissingX_->getBinContent(i))).first);
      meEfficiencyX_->setBinError(i,(eff(meValidX_->getBinContent(i),meMissingX_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidY_->getNbinsX();++i){
      meEfficiencyY_->setBinContent(i,(eff(meValidY_->getBinContent(i),meMissingY_->getBinContent(i))).first);
      meEfficiencyY_->setBinError(i,(eff(meValidY_->getBinContent(i),meMissingY_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidAlpha_->getNbinsX();++i){
      meEfficiencyAlpha_->setBinContent(i,(eff(meValidAlpha_->getBinContent(i),meMissingAlpha_->getBinContent(i))).first);
      meEfficiencyAlpha_->setBinError(i,(eff(meValidAlpha_->getBinContent(i),meMissingAlpha_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidBeta_->getNbinsX();++i){
      meEfficiencyBeta_->setBinContent(i,(eff(meValidBeta_->getBinContent(i),meMissingBeta_->getBinContent(i))).first);
      meEfficiencyBeta_->setBinError(i,(eff(meValidBeta_->getBinContent(i),meMissingBeta_->getBinContent(i))).second);
    }*/
  }
  if(ladon && barrel){
    meEfficiencyLad_->setBinContent(1,(eff(meValidLad_->getBinContent(1),meMissingLad_->getBinContent(1))).first);
    meEfficiencyLad_->setBinError(1,(eff(meValidLad_->getBinContent(1),meMissingLad_->getBinContent(1))).second);
    for(int i=1;i<=meValidXLad_->getNbinsX();++i){
      meEfficiencyXLad_->setBinContent(i,(eff(meValidXLad_->getBinContent(i),meMissingXLad_->getBinContent(i))).first);
      meEfficiencyXLad_->setBinError(i,(eff(meValidXLad_->getBinContent(i),meMissingXLad_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidYLad_->getNbinsX();++i){
      meEfficiencyYLad_->setBinContent(i,(eff(meValidYLad_->getBinContent(i),meMissingYLad_->getBinContent(i))).first);
      meEfficiencyYLad_->setBinError(i,(eff(meValidYLad_->getBinContent(i),meMissingYLad_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidAlphaLad_->getNbinsX();++i){
      meEfficiencyAlphaLad_->setBinContent(i,(eff(meValidAlphaLad_->getBinContent(i),meMissingAlphaLad_->getBinContent(i))).first);
      meEfficiencyAlphaLad_->setBinError(i,(eff(meValidAlphaLad_->getBinContent(i),meMissingAlphaLad_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidBetaLad_->getNbinsX();++i){
      meEfficiencyBetaLad_->setBinContent(i,(eff(meValidBetaLad_->getBinContent(i),meMissingBetaLad_->getBinContent(i))).first);
      meEfficiencyBetaLad_->setBinError(i,(eff(meValidBetaLad_->getBinContent(i),meMissingBetaLad_->getBinContent(i))).second);
    }
  }
  
  if(layon && barrel){
    meEfficiencyLay_->setBinContent(1,(eff(meValidLay_->getBinContent(1),meMissingLay_->getBinContent(1))).first);
    meEfficiencyLay_->setBinError(1,(eff(meValidLay_->getBinContent(1),meMissingLay_->getBinContent(1))).second);
    for(int i=1;i<=meValidXLay_->getNbinsX();++i){
      meEfficiencyXLay_->setBinContent(i,(eff(meValidXLay_->getBinContent(i),meMissingXLay_->getBinContent(i))).first);
      meEfficiencyXLay_->setBinError(i,(eff(meValidXLay_->getBinContent(i),meMissingXLay_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidYLay_->getNbinsX();++i){
      meEfficiencyYLay_->setBinContent(i,(eff(meValidYLay_->getBinContent(i),meMissingYLay_->getBinContent(i))).first);
      meEfficiencyYLay_->setBinError(i,(eff(meValidYLay_->getBinContent(i),meMissingYLay_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidAlphaLay_->getNbinsX();++i){
      meEfficiencyAlphaLay_->setBinContent(i,(eff(meValidAlphaLay_->getBinContent(i),meMissingAlphaLay_->getBinContent(i))).first);
      meEfficiencyAlphaLay_->setBinError(i,(eff(meValidAlphaLay_->getBinContent(i),meMissingAlphaLay_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidBetaLay_->getNbinsX();++i){
      meEfficiencyBetaLay_->setBinContent(i,(eff(meValidBetaLay_->getBinContent(i),meMissingBetaLay_->getBinContent(i))).first);
      meEfficiencyBetaLay_->setBinError(i,(eff(meValidBetaLay_->getBinContent(i),meMissingBetaLay_->getBinContent(i))).second);
    }
  }
  
  if(phion && barrel){
    meEfficiencyPhi_->setBinContent(1,(eff(meValidPhi_->getBinContent(1),meMissingPhi_->getBinContent(1))).first);
    meEfficiencyPhi_->setBinError(1,(eff(meValidPhi_->getBinContent(1),meMissingPhi_->getBinContent(1))).second);
    for(int i=1;i<=meValidXPhi_->getNbinsX();++i){
      meEfficiencyXPhi_->setBinContent(i,(eff(meValidXPhi_->getBinContent(i),meMissingXPhi_->getBinContent(i))).first);
      meEfficiencyXPhi_->setBinError(i,(eff(meValidXPhi_->getBinContent(i),meMissingXPhi_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidYPhi_->getNbinsX();++i){
      meEfficiencyYPhi_->setBinContent(i,(eff(meValidYPhi_->getBinContent(i),meMissingYPhi_->getBinContent(i))).first);
      meEfficiencyYPhi_->setBinError(i,(eff(meValidYPhi_->getBinContent(i),meMissingYPhi_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidAlphaPhi_->getNbinsX();++i){
      meEfficiencyAlphaPhi_->setBinContent(i,(eff(meValidAlphaPhi_->getBinContent(i),meMissingAlphaPhi_->getBinContent(i))).first);
      meEfficiencyAlphaPhi_->setBinError(i,(eff(meValidAlphaPhi_->getBinContent(i),meMissingAlphaPhi_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidBetaPhi_->getNbinsX();++i){
      meEfficiencyBetaPhi_->setBinContent(i,(eff(meValidBetaPhi_->getBinContent(i),meMissingBetaPhi_->getBinContent(i))).first);
      meEfficiencyBetaPhi_->setBinError(i,(eff(meValidBetaPhi_->getBinContent(i),meMissingBetaPhi_->getBinContent(i))).second);
    }
  }
  if(bladeon && endcap){
    meEfficiencyBlade_->setBinContent(1,(eff(meValidBlade_->getBinContent(1),meMissingBlade_->getBinContent(1))).first);
    meEfficiencyBlade_->setBinError(1,(eff(meValidBlade_->getBinContent(1),meMissingBlade_->getBinContent(1))).second);
    for(int i=1;i<=meValidXBlade_->getNbinsX();++i){
      meEfficiencyXBlade_->setBinContent(i,(eff(meValidXBlade_->getBinContent(i),meMissingXBlade_->getBinContent(i))).first);
      meEfficiencyXBlade_->setBinError(i,(eff(meValidXBlade_->getBinContent(i),meMissingXBlade_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidYBlade_->getNbinsX();++i){
      meEfficiencyYBlade_->setBinContent(i,(eff(meValidYBlade_->getBinContent(i),meMissingYBlade_->getBinContent(i))).first);
      meEfficiencyYBlade_->setBinError(i,(eff(meValidYBlade_->getBinContent(i),meMissingYBlade_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidAlphaBlade_->getNbinsX();++i){
      meEfficiencyAlphaBlade_->setBinContent(i,(eff(meValidAlphaBlade_->getBinContent(i),meMissingAlphaBlade_->getBinContent(i))).first);
      meEfficiencyAlphaBlade_->setBinError(i,(eff(meValidAlphaBlade_->getBinContent(i),meMissingAlphaBlade_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidBetaBlade_->getNbinsX();++i){
      meEfficiencyBetaBlade_->setBinContent(i,(eff(meValidBetaBlade_->getBinContent(i),meMissingBetaBlade_->getBinContent(i))).first);
      meEfficiencyBetaBlade_->setBinError(i,(eff(meValidBetaBlade_->getBinContent(i),meMissingBetaBlade_->getBinContent(i))).second);
    }
  }
  if(diskon && endcap){
    meEfficiencyDisk_->setBinContent(1,(eff(meValidDisk_->getBinContent(1),meMissingDisk_->getBinContent(1))).first);
    meEfficiencyDisk_->setBinError(1,(eff(meValidDisk_->getBinContent(1),meMissingDisk_->getBinContent(1))).second);
    for(int i=1;i<=meValidXDisk_->getNbinsX();++i){
      meEfficiencyXDisk_->setBinContent(i,(eff(meValidXDisk_->getBinContent(i),meMissingXDisk_->getBinContent(i))).first);
      meEfficiencyXDisk_->setBinError(i,(eff(meValidXDisk_->getBinContent(i),meMissingXDisk_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidYDisk_->getNbinsX();++i){
      meEfficiencyYDisk_->setBinContent(i,(eff(meValidYDisk_->getBinContent(i),meMissingYDisk_->getBinContent(i))).first);
      meEfficiencyYDisk_->setBinError(i,(eff(meValidYDisk_->getBinContent(i),meMissingYDisk_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidAlphaDisk_->getNbinsX();++i){
      meEfficiencyAlphaDisk_->setBinContent(i,(eff(meValidAlphaDisk_->getBinContent(i),meMissingAlphaDisk_->getBinContent(i))).first);
      meEfficiencyAlphaDisk_->setBinError(i,(eff(meValidAlphaDisk_->getBinContent(i),meMissingAlphaDisk_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidBetaDisk_->getNbinsX();++i){
      meEfficiencyBetaDisk_->setBinContent(i,(eff(meValidBetaDisk_->getBinContent(i),meMissingBetaDisk_->getBinContent(i))).first);
      meEfficiencyBetaDisk_->setBinError(i,(eff(meValidBetaDisk_->getBinContent(i),meMissingBetaDisk_->getBinContent(i))).second);
    }
  }
  if(ringon && endcap){
    meEfficiencyRing_->setBinContent(1,(eff(meValidRing_->getBinContent(1),meMissingRing_->getBinContent(1))).first);
    meEfficiencyRing_->setBinError(1,(eff(meValidRing_->getBinContent(1),meMissingRing_->getBinContent(1))).second);
    for(int i=1;i<=meValidXRing_->getNbinsX();++i){
      meEfficiencyXRing_->setBinContent(i,(eff(meValidXRing_->getBinContent(i),meMissingXRing_->getBinContent(i))).first);
      meEfficiencyXRing_->setBinError(i,(eff(meValidXRing_->getBinContent(i),meMissingXRing_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidYRing_->getNbinsX();++i){
      meEfficiencyYRing_->setBinContent(i,(eff(meValidYRing_->getBinContent(i),meMissingYRing_->getBinContent(i))).first);
      meEfficiencyYRing_->setBinError(i,(eff(meValidYRing_->getBinContent(i),meMissingYRing_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidAlphaRing_->getNbinsX();++i){
      meEfficiencyAlphaRing_->setBinContent(i,(eff(meValidAlphaRing_->getBinContent(i),meMissingAlphaRing_->getBinContent(i))).first);
      meEfficiencyAlphaRing_->setBinError(i,(eff(meValidAlphaRing_->getBinContent(i),meMissingAlphaRing_->getBinContent(i))).second);
    }
    for(int i=1;i<=meValidBetaRing_->getNbinsX();++i){
      meEfficiencyBetaRing_->setBinContent(i,(eff(meValidBetaRing_->getBinContent(i),meMissingBetaRing_->getBinContent(i))).first);
      meEfficiencyBetaRing_->setBinError(i,(eff(meValidBetaRing_->getBinContent(i),meMissingBetaRing_->getBinContent(i))).second);
    }
  }

}

std::pair<double,double> SiPixelHitEfficiencyModule::eff(double nValid, double nMissing){
  double efficiency = 0, error = 0 ;
  if(nValid+nMissing!=0){
    efficiency=nValid/(nValid+nMissing);
    error=sqrt(efficiency*(1.-efficiency)/(nValid+nMissing));
  }
  return make_pair(efficiency,error);
}
