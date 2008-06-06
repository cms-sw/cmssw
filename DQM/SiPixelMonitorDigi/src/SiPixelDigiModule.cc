#include "DQM/SiPixelMonitorDigi/interface/SiPixelDigiModule.h"
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
SiPixelDigiModule::SiPixelDigiModule() : id_(0),
					 ncols_(416),
					 nrows_(160) 
{
}
///
SiPixelDigiModule::SiPixelDigiModule(const uint32_t& id) : 
  id_(id),
  ncols_(416),
  nrows_(160)
{ 
}
///
SiPixelDigiModule::SiPixelDigiModule(const uint32_t& id, const int& ncols, const int& nrows) : 
  id_(id),
  ncols_(ncols),
  nrows_(nrows)
{ 
}
//
// Destructor
//
SiPixelDigiModule::~SiPixelDigiModule() {}
//
// Book histograms
//
void SiPixelDigiModule::book(const edm::ParameterSet& iConfig, int type) {
  bool barrel = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  bool isHalfModule = false;
  if(barrel){
    isHalfModule = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).isHalfModule(); 
  }

  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = iConfig.getParameter<edm::InputTag>( "src" );
  

  // Get DQM interface
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
  
  int nbinx = ncols_/2;
  int nbiny = nrows_/2;


  if(type==0){
    SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );
    // Number of digis
    hid = theHistogramId->setHistoId("ndigis",id_);
    meNDigis_ = theDMBE->book1D(hid,"Number of Digis",50,0.,50.);
    meNDigis_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
    hid = theHistogramId->setHistoId("adc",id_);
    meADC_ = theDMBE->book1D(hid,"Digi charge",500,0.,500.);
    meADC_->setAxisTitle("ADC counts",1);
    // 2D hit map
    hid = theHistogramId->setHistoId("hitmap",id_);
    mePixDigis_ = theDMBE->book2D(hid,"Number of Digis (1bin=four pixels)",nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
    mePixDigis_->setAxisTitle("Columns",1);
    mePixDigis_->setAxisTitle("Rows",2);
    delete theHistogramId;
  }
  
  if(type==1 && barrel){
    uint32_t DBladder = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).ladderName();
    char sladder[80]; sprintf(sladder,"Ladder_%02i",DBladder);
    hid = src.label() + "_" + sladder;
    if(isHalfModule) hid += "H";
    else hid += "F";
    // Number of digis
    meNDigisLad_ = theDMBE->book1D("ndigis_"+hid,"Number of Digis",50,0.,50.);
    meNDigisLad_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
    meADCLad_ = theDMBE->book1D("adc_" + hid,"Digi charge",500,0.,500.);
    meADCLad_->setAxisTitle("ADC counts",1);
    // 2D hit map
    mePixDigisLad_ = theDMBE->book2D("hitmap_"+hid,"Number of Digis (1bin=four pixels)",nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
    mePixDigisLad_->setAxisTitle("Columns",1);
    mePixDigisLad_->setAxisTitle("Rows",2);
  }
  if(type==2 && barrel){
    uint32_t DBlayer = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).layerName();
    char slayer[80]; sprintf(slayer,"Layer_%i",DBlayer);
    hid = src.label() + "_" + slayer;
    // Number of digis
    meNDigisLay_ = theDMBE->book1D("ndigis_"+hid,"Number of Digis",50,0.,50.);
    meNDigisLay_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
    meADCLay_ = theDMBE->book1D("adc_" + hid,"Digi charge",500,0.,500.);
    meADCLay_->setAxisTitle("ADC counts",1);
    // 2D hit map
    if(isHalfModule){
      mePixDigisLay_ = theDMBE->book2D("hitmap_"+hid,"Number of Digis (1bin=four pixels)",nbinx,0.,float(ncols_),2*nbiny,0.,float(2*nrows_));
    }
    else{
      mePixDigisLay_ = theDMBE->book2D("hitmap_"+hid,"Number of Digis (1bin=four pixels)",nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
    }
    mePixDigisLay_->setAxisTitle("Columns",1);
    mePixDigisLay_->setAxisTitle("Rows",2);
  }
  if(type==3 && barrel){
    uint32_t DBmodule = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).moduleName();
    char smodule[80]; sprintf(smodule,"Ring_%i",DBmodule);
    hid = src.label() + "_" + smodule;
    // Number of digis
    meNDigisPhi_ = theDMBE->book1D("ndigis_"+hid,"Number of Digis",50,0.,50.);
    meNDigisPhi_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
    meADCPhi_ = theDMBE->book1D("adc_" + hid,"Digi charge",500,0.,500.);
    meADCPhi_->setAxisTitle("ADC counts",1);
    // 2D hit map
    if(isHalfModule){
      mePixDigisPhi_ = theDMBE->book2D("hitmap_"+hid,"Number of Digis (1bin=four pixels)",nbinx,0.,float(ncols_),2*nbiny,0.,float(2*nrows_));
    }
    else {
      mePixDigisPhi_ = theDMBE->book2D("hitmap_"+hid,"Number of Digis (1bin=four pixels)",nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
    }
    mePixDigisPhi_->setAxisTitle("Columns",1);
    mePixDigisPhi_->setAxisTitle("Rows",2);
  }
  if(type==4 && endcap){
    uint32_t blade= PixelEndcapName::PixelEndcapName(DetId::DetId(id_)).bladeName();
    
    char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
    hid = src.label() + "_" + sblade;
    // Number of digis
    meNDigisBlade_ = theDMBE->book1D("ndigis_"+hid,"Number of Digis",50,0.,50.);
    meNDigisBlade_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
    meADCBlade_ = theDMBE->book1D("adc_" + hid,"Digi charge",500,0.,500.);
    meADCBlade_->setAxisTitle("ADC counts",1);
  }
  if(type==5 && endcap){
    uint32_t disk = PixelEndcapName::PixelEndcapName(DetId::DetId(id_)).diskName();
    
    char sdisk[80]; sprintf(sdisk, "Disk_%i",disk);
    hid = src.label() + "_" + sdisk;
    // Number of digis
    meNDigisDisk_ = theDMBE->book1D("ndigis_"+hid,"Number of Digis",50,0.,50.);
    meNDigisDisk_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
    meADCDisk_ = theDMBE->book1D("adc_" + hid,"Digi charge",500,0.,500.);
    meADCDisk_->setAxisTitle("ADC counts",1);
  }
  if(type==6 && endcap){
    uint32_t panel= PixelEndcapName::PixelEndcapName(DetId::DetId(id_)).pannelName();
    uint32_t module= PixelEndcapName::PixelEndcapName(DetId::DetId(id_)).plaquetteName();
    char slab[80]; sprintf(slab, "Panel_%i_Ring_%i",panel, module);
    hid = src.label() + "_" + slab;
    // Number of digis
    meNDigisRing_ = theDMBE->book1D("ndigis_"+hid,"Number of Digis",50,0.,50.);
    meNDigisRing_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
    meADCRing_ = theDMBE->book1D("adc_" + hid,"Digi charge",500,0.,500.);
    meADCRing_->setAxisTitle("ADC counts",1);
    // 2D hit map
    mePixDigisRing_ = theDMBE->book2D("hitmap_"+hid,"Number of Digis (1bin=four pixels)",nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
    mePixDigisRing_->setAxisTitle("Columns",1);
    mePixDigisRing_->setAxisTitle("Rows",2);
  }
}


//
// Fill histograms
//
void SiPixelDigiModule::fill(const edm::DetSetVector<PixelDigi>& input, bool modon, bool ladon, bool layon, bool phion, bool bladeon, bool diskon, bool ringon) {
  bool barrel = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId::DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  bool isHalfModule = false;
  uint32_t DBladder = 0;
  if(barrel){
    isHalfModule = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).isHalfModule(); 
    DBladder = PixelBarrelName::PixelBarrelName(DetId::DetId(id_)).ladderName();
  }

  // Get DQM interface
  //DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
  //std::cout<<"id_ = "<<id_<<" , dmbe="<<theDMBE->pwd()<<std::endl;
  //std::cout<<"********************"<<std::endl;
  edm::DetSetVector<PixelDigi>::const_iterator isearch = input.find(id_); // search  digis of detid
  
  if( isearch != input.end() ) {  // Not an empty iterator
    
    unsigned int numberOfDigis = 0;
    
    // Look at digis now
    edm::DetSet<PixelDigi>::const_iterator  di;
    for(di = isearch->data.begin(); di != isearch->data.end(); di++) {
      numberOfDigis++;
      int adc = di->adc();    // charge
      int col = di->column(); // column 
      int row = di->row();    // row
      if(modon){
	(mePixDigis_)->Fill((float)col,(float)row);
	(meADC_)->Fill((float)adc);
      }
      if(ladon && barrel){
	(mePixDigisLad_)->Fill((float)col,(float)row);
	(meADCLad_)->Fill((float)adc);
      }
      if(layon && barrel){
	if(isHalfModule && DBladder==1){
	  (mePixDigisLay_)->Fill((float)col,(float)row+80);
	}
	else (mePixDigisLay_)->Fill((float)col,(float)row);
	(meADCLay_)->Fill((float)adc);
      }
      if(phion && barrel){
	if(isHalfModule && DBladder==1){
	  (mePixDigisPhi_)->Fill((float)col,(float)row+80);
	}
	else (mePixDigisPhi_)->Fill((float)col,(float)row);
	(meADCPhi_)->Fill((float)adc);
      }
      if(bladeon && endcap){
	(meADCBlade_)->Fill((float)adc);
      }
      if(diskon && endcap){
	(meADCDisk_)->Fill((float)adc);
      }
      if(ringon && endcap){
	(mePixDigisRing_)->Fill((float)col,(float)row);
	(meADCRing_)->Fill((float)adc);
      }
    }
    if(modon) (meNDigis_)->Fill((float)numberOfDigis);
    if(ladon && barrel) (meNDigisLad_)->Fill((float)numberOfDigis);
    if(layon && barrel) (meNDigisLay_)->Fill((float)numberOfDigis);
    if(phion && barrel) (meNDigisPhi_)->Fill((float)numberOfDigis);
    if(bladeon && endcap) (meNDigisBlade_)->Fill((float)numberOfDigis);
    if(diskon && endcap) (meNDigisDisk_)->Fill((float)numberOfDigis);
    if(ringon && endcap) (meNDigisRing_)->Fill((float)numberOfDigis);
    //std::cout<<"number of digis="<<numberOfDigis<<std::endl;
   
  }
  
  
  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;
  
}
