#include "DQM/SiPixelMonitorDigi/interface/SiPixelDigiModule.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
/// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <stdlib.h>
//
// Constructors
//
SiPixelDigiModule::SiPixelDigiModule() : id_(0),
					 ncols_(416),
					 nrows_(160) 
{

  bookedUL_ = false;

}
///
SiPixelDigiModule::SiPixelDigiModule(const uint32_t& id) : 
  id_(id),
  ncols_(416),
  nrows_(160)
{ 

  bookedUL_ = false;

}
///
SiPixelDigiModule::SiPixelDigiModule(const uint32_t& id, const int& ncols, const int& nrows) : 
  id_(id),
  ncols_(ncols),
  nrows_(nrows)
{ 

  bookedUL_ = false;

}
//
// Destructor
//
SiPixelDigiModule::~SiPixelDigiModule() {}
//
// Book histograms
//
void SiPixelDigiModule::book(const edm::ParameterSet& iConfig) {

  bool modon = iConfig.getParameter<bool>("Mod_On");
  
  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = iConfig.getParameter<edm::InputTag>( "src" );
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );
  // Get DQM interface
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
  if(!bookedUL_) bookUpperLevelMEs(theDMBE);
  
  if(modon){
    // Number of digis
    hid = theHistogramId->setHistoId("ndigis",id_);
    //std::cout << hid << " " << theHistogramId->getDataCollection(hid) << " " << theHistogramId->getRawId(hid) << std::endl;
    meNDigis_ = theDMBE->book1D(hid,"Number of Digis",50,0.,50.);
    meNDigis_->setAxisTitle("Number of digis",1);
   // Charge in ADC counts
    hid = theHistogramId->setHistoId("adc",id_);
    meADC_ = theDMBE->book1D(hid,"Digi charge",500,0.,500.);
    meADC_->setAxisTitle("ADC counts",1);
   // 2D hit map
    int nbinx = ncols_/2;
    int nbiny = nrows_/2;
    hid = theHistogramId->setHistoId("hitmap",id_);
    mePixDigis_ = theDMBE->book2D(hid,"Number of Digis (1bin=four pixels)",nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
    mePixDigis_->setAxisTitle("Columns",1);
    mePixDigis_->setAxisTitle("Rows",2);
  }
  
  delete theHistogramId;
  
}

void SiPixelDigiModule::bookUpperLevelMEs(DQMStore* theDMBE){
  std::string theDir = theDMBE->pwd();
  theDMBE->cd();
  bookUpperLevelBarrelMEs(theDMBE);
  theDMBE->cd();
  bookUpperLevelEndcapMEs(theDMBE);
  theDMBE->cd();
  theDMBE->cd(theDir);
  bookedUL_ = true;
}

void SiPixelDigiModule::bookUpperLevelBarrelMEs(DQMStore* theDMBE){
  
  std::string currDir = theDMBE->pwd();
  if(currDir.find("Shell_mI/Layer_1")!=std::string::npos){
    Barrel_SmIL1_ndigis = theDMBE->book1D("Barrel_SmIL1_ndigis","Number of Digis in BarrelSmIL1",50,0.,50.);
    Barrel_SmIL1_ndigis->setAxisTitle("Number of digis",1);
    Barrel_SmIL1_adc = theDMBE->book1D("Barrel_SmIL1_adc","Digi charge in BarrelSmIL1",500,0.,500.);
    Barrel_SmIL1_adc->setAxisTitle("ADC counts",1);
    Barrel_SmIL1_hitmap = theDMBE->book2D("Barrel_SmIL1_hitmap","Occupancy in BarrelSmIL1",208,0.,416.,80,0.,80.);
    Barrel_SmIL1_hitmap->setAxisTitle("Columns",1);
    Barrel_SmIL1_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("Shell_mI/Layer_2")!=std::string::npos){
    Barrel_SmIL2_ndigis = theDMBE->book1D("Barrel_SmIL2_ndigis","Number of Digis in BarrelSmIL2",50,0.,50.);
    Barrel_SmIL2_ndigis->setAxisTitle("Number of digis",1);
    Barrel_SmIL2_adc = theDMBE->book1D("Barrel_SmIL2_adc","Digi charge in BarrelSmIL2",500,0.,500.);
    Barrel_SmIL2_adc->setAxisTitle("ADC counts",1);
    Barrel_SmIL2_hitmap = theDMBE->book2D("Barrel_SmIL2_hitmap","Occupancy in BarrelSmIL2",208,0.,416.,80,0.,80.);
    Barrel_SmIL2_hitmap->setAxisTitle("Columns",1);
    Barrel_SmIL2_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("Shell_mI/Layer_3")!=std::string::npos){
    Barrel_SmIL3_ndigis = theDMBE->book1D("Barrel_SmIL3_ndigis","Number of Digis in BarrelSmIL3",50,0.,50.);
    Barrel_SmIL3_ndigis->setAxisTitle("Number of digis",1);
    Barrel_SmIL3_adc = theDMBE->book1D("Barrel_SmIL3_adc","Digi charge in BarrelSmIL3",500,0.,500.);
    Barrel_SmIL3_adc->setAxisTitle("ADC counts",1);
    Barrel_SmIL3_hitmap = theDMBE->book2D("Barrel_SmIL3_hitmap","Occupancy in BarrelSmIL3",208,0.,416.,80,0.,80.);
    Barrel_SmIL3_hitmap->setAxisTitle("Columns",1);
    Barrel_SmIL3_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("Shell_mO/Layer_1")!=std::string::npos){
    Barrel_SmOL1_ndigis = theDMBE->book1D("Barrel_SmOL1_ndigis","Number of Digis in BarrelSmOL1",50,0.,50.);
    Barrel_SmOL1_ndigis->setAxisTitle("Number of digis",1);
    Barrel_SmOL1_adc = theDMBE->book1D("Barrel_SmOL1_adc","Digi charge in BarrelSmOL1",500,0.,500.);
    Barrel_SmOL1_adc->setAxisTitle("ADC counts",1);
    Barrel_SmOL1_hitmap = theDMBE->book2D("Barrel_SmOL1_hitmap","Occupancy in BarrelSmOL1",208,0.,416.,80,0.,80.);
    Barrel_SmOL1_hitmap->setAxisTitle("Columns",1);
    Barrel_SmOL1_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("Shell_mO/Layer_2")!=std::string::npos){
    Barrel_SmOL2_ndigis = theDMBE->book1D("Barrel_SmOL2_ndigis","Number of Digis in BarrelSmOL2",50,0.,50.);
    Barrel_SmOL2_ndigis->setAxisTitle("Number of digis",1);
    Barrel_SmOL2_adc = theDMBE->book1D("Barrel_SmOL2_adc","Digi charge in BarrelSmOL2",500,0.,500.);
    Barrel_SmOL2_adc->setAxisTitle("ADC counts",1);
    Barrel_SmOL2_hitmap = theDMBE->book2D("Barrel_SmOL2_hitmap","Occupancy in BarrelSmOL2",208,0.,416.,80,0.,80.);
    Barrel_SmOL2_hitmap->setAxisTitle("Columns",1);
    Barrel_SmOL2_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("Shell_mO/Layer_3")!=std::string::npos){
    Barrel_SmOL3_ndigis = theDMBE->book1D("Barrel_SmOL3_ndigis","Number of Digis in BarrelSmOL3",50,0.,50.);
    Barrel_SmOL3_ndigis->setAxisTitle("Number of digis",1);
    Barrel_SmOL3_adc = theDMBE->book1D("Barrel_SmOL3_adc","Digi charge in BarrelSmOL3",500,0.,500.);
    Barrel_SmOL3_adc->setAxisTitle("ADC counts",1);
    Barrel_SmOL3_hitmap = theDMBE->book2D("Barrel_SmOL3_hitmap","Occupancy in BarrelSmOL3",208,0.,416.,80,0.,80.);
    Barrel_SmOL3_hitmap->setAxisTitle("Columns",1);
    Barrel_SmOL3_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("Shell_pI/Layer_1")!=std::string::npos){
    Barrel_SpIL1_ndigis = theDMBE->book1D("Barrel_SpIL1_ndigis","Number of Digis in BarrelSpIL1",50,0.,50.);
    Barrel_SpIL1_ndigis->setAxisTitle("Number of digis",1);
    Barrel_SpIL1_adc = theDMBE->book1D("Barrel_SpIL1_adc","Digi charge in BarrelSpIL1",500,0.,500.);
    Barrel_SpIL1_adc->setAxisTitle("ADC counts",1);
    Barrel_SpIL1_hitmap = theDMBE->book2D("Barrel_SpIL1_hitmap","Occupancy in BarrelSpIL1",208,0.,416.,80,0.,80.);
    Barrel_SpIL1_hitmap->setAxisTitle("Columns",1);
    Barrel_SpIL1_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("Shell_pI/Layer_2")!=std::string::npos){
    Barrel_SpIL2_ndigis = theDMBE->book1D("Barrel_SpIL2_ndigis","Number of Digis in BarrelSpIL2",50,0.,50.);
    Barrel_SpIL2_ndigis->setAxisTitle("Number of digis",1);
    Barrel_SpIL2_adc = theDMBE->book1D("Barrel_SpIL2_adc","Digi charge in BarrelSpIL2",500,0.,500.);
    Barrel_SpIL2_adc->setAxisTitle("ADC counts",1);
    Barrel_SpIL2_hitmap = theDMBE->book2D("Barrel_SpIL2_hitmap","Occupancy in BarrelSpIL2",208,0.,416.,80,0.,80.);
    Barrel_SpIL2_hitmap->setAxisTitle("Columns",1);
    Barrel_SpIL2_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("Shell_pI/Layer_3")!=std::string::npos){
    Barrel_SpIL3_ndigis = theDMBE->book1D("Barrel_SpIL3_ndigis","Number of Digis in BarrelSpIL3",50,0.,50.);
    Barrel_SpIL3_ndigis->setAxisTitle("Number of digis",1);
    Barrel_SpIL3_adc = theDMBE->book1D("Barrel_SpIL3_adc","Digi charge in BarrelSpIL3",500,0.,500.);
    Barrel_SpIL3_adc->setAxisTitle("ADC counts",1);
    Barrel_SpIL3_hitmap = theDMBE->book2D("Barrel_SpIL3_hitmap","Occupancy in BarrelSpIL3",208,0.,416.,80,0.,80.);
    Barrel_SpIL3_hitmap->setAxisTitle("Columns",1);
    Barrel_SpIL3_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("Shell_pO/Layer_1")!=std::string::npos){
    Barrel_SpOL1_ndigis = theDMBE->book1D("Barrel_SpOL1_ndigis","Number of Digis in BarrelSpOL1",50,0.,50.);
    Barrel_SpOL1_ndigis->setAxisTitle("Number of digis",1);
    Barrel_SpOL1_adc = theDMBE->book1D("Barrel_SpOL1_adc","Digi charge in BarrelSpOL1",500,0.,500.);
    Barrel_SpOL1_adc->setAxisTitle("ADC counts",1);
    Barrel_SpOL1_hitmap = theDMBE->book2D("Barrel_SpOL1_hitmap","Occupancy in BarrelSpOL1",208,0.,416.,80,0.,80.);
    Barrel_SpOL1_hitmap->setAxisTitle("Columns",1);
    Barrel_SpOL1_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("Shell_pO/Layer_2")!=std::string::npos){
    Barrel_SpOL2_ndigis = theDMBE->book1D("Barrel_SpOL2_ndigis","Number of Digis in BarrelSpOL2",50,0.,50.);
    Barrel_SpOL2_ndigis->setAxisTitle("Number of digis",1);
    Barrel_SpOL2_adc = theDMBE->book1D("Barrel_SpOL2_adc","Digi charge in BarrelSpOL2",500,0.,500.);
    Barrel_SpOL2_adc->setAxisTitle("ADC counts",1);
    Barrel_SpOL2_hitmap = theDMBE->book2D("Barrel_SpOL2_hitmap","Occupancy in BarrelSpOL2",208,0.,416.,80,0.,80.);
    Barrel_SpOL2_hitmap->setAxisTitle("Columns",1);
    Barrel_SpOL2_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("Shell_pO/Layer_3")!=std::string::npos){
    Barrel_SpOL3_ndigis = theDMBE->book1D("Barrel_SpOL3_ndigis","Number of Digis in BarrelSpOL3",50,0.,50.);
    Barrel_SpOL3_ndigis->setAxisTitle("Number of digis",1);
    Barrel_SpOL3_adc = theDMBE->book1D("Barrel_SpOL3_adc","Digi charge in BarrelSpOL3",500,0.,500.);
    Barrel_SpOL3_adc->setAxisTitle("ADC counts",1);
    Barrel_SpOL3_hitmap = theDMBE->book2D("Barrel_SpOL3_hitmap","Occupancy in BarrelSpOL3",208,0.,416.,80,0.,80.);
    Barrel_SpOL3_hitmap->setAxisTitle("Columns",1);
    Barrel_SpOL3_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else{
    std::vector<std::string> subdirs = theDMBE->getSubdirs();
    for(std::vector<std::string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++){
      if((theDMBE->pwd()).find("Endcap")!=std::string::npos) theDMBE->goUp();
      theDMBE->cd(*it);
      if((theDMBE->pwd()).find("Endcap")!=std::string::npos) continue;
      bookUpperLevelBarrelMEs(theDMBE);
      theDMBE->goUp();
    }
  }

}


void SiPixelDigiModule::bookUpperLevelEndcapMEs(DQMStore* theDMBE){
  
  std::string currDir = theDMBE->pwd();
  if(currDir.find("HalfCylinder_mI/Disk_1")!=std::string::npos){
    Endcap_HCmID1_ndigis = theDMBE->book1D("Endcap_HCmID1_ndigis","Number of Digis in EndcapHCmID1",50,0.,50.);
    Endcap_HCmID1_ndigis->setAxisTitle("Number of digis",1);
    Endcap_HCmID1_adc = theDMBE->book1D("Endcap_HCmID1_adc","Digi charge in EndcapHCmID1",500,0.,500.);
    Endcap_HCmID1_adc->setAxisTitle("ADC counts",1);
    Endcap_HCmID1_hitmap = theDMBE->book2D("Endcap_HCmID1_hitmap","Occupancy in EndcapHCmID1",208,0.,416.,80,0.,80.);
    Endcap_HCmID1_hitmap->setAxisTitle("Columns",1);
    Endcap_HCmID1_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("HalfCylinder_mI/Disk_2")!=std::string::npos){
    Endcap_HCmID2_ndigis = theDMBE->book1D("Endcap_HCmID2_ndigis","Number of Digis in EndcapHCmID2",50,0.,50.);
    Endcap_HCmID2_ndigis->setAxisTitle("Number of digis",1);
    Endcap_HCmID2_adc = theDMBE->book1D("Endcap_HCmID2_adc","Digi charge in EndcapHCmID2",500,0.,500.);
    Endcap_HCmID2_adc->setAxisTitle("ADC counts",1);
    Endcap_HCmID2_hitmap = theDMBE->book2D("Endcap_HCmID2_hitmap","Occupancy in EndcapHCmID2",208,0.,416.,80,0.,80.);
    Endcap_HCmID2_hitmap->setAxisTitle("Columns",1);
    Endcap_HCmID2_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("HalfCylinder_mO/Disk_1")!=std::string::npos){
    Endcap_HCmOD1_ndigis = theDMBE->book1D("Endcap_HCmOD1_ndigis","Number of Digis in EndcapHCmOD1",50,0.,50.);
    Endcap_HCmOD1_ndigis->setAxisTitle("Number of digis",1);
    Endcap_HCmOD1_adc = theDMBE->book1D("Endcap_HCmOD1_adc","Digi charge in EndcapHCmOD1",500,0.,500.);
    Endcap_HCmOD1_adc->setAxisTitle("ADC counts",1);
    Endcap_HCmOD1_hitmap = theDMBE->book2D("Endcap_HCmOD1_hitmap","Occupancy in EndcapHCmOD1",208,0.,416.,80,0.,80.);
    Endcap_HCmOD1_hitmap->setAxisTitle("Columns",1);
    Endcap_HCmOD1_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("HalfCylinder_mO/Disk_2")!=std::string::npos){
    Endcap_HCmOD2_ndigis = theDMBE->book1D("Endcap_HCmOD2_ndigis","Number of Digis in EndcapHCmOD2",50,0.,50.);
    Endcap_HCmOD2_ndigis->setAxisTitle("Number of digis",1);
    Endcap_HCmOD2_adc = theDMBE->book1D("Endcap_HCmOD2_adc","Digi charge in EndcapHCmOD2",500,0.,500.);
    Endcap_HCmOD2_adc->setAxisTitle("ADC counts",1);
    Endcap_HCmOD2_hitmap = theDMBE->book2D("Endcap_HCmOD2_hitmap","Occupancy in EndcapHCmOD2",208,0.,416.,80,0.,80.);
    Endcap_HCmOD2_hitmap->setAxisTitle("Columns",1);
    Endcap_HCmOD2_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("HalfCylinder_pI/Disk_1")!=std::string::npos){
    Endcap_HCpID1_ndigis = theDMBE->book1D("Endcap_HCpID1_ndigis","Number of Digis in EndcapHCpID1",50,0.,50.);
    Endcap_HCpID1_ndigis->setAxisTitle("Number of digis",1);
    Endcap_HCpID1_adc = theDMBE->book1D("Endcap_HCpID1_adc","Digi charge in EndcapHCpID1",500,0.,500.);
    Endcap_HCpID1_adc->setAxisTitle("ADC counts",1);
    Endcap_HCpID1_hitmap = theDMBE->book2D("Endcap_HCpID1_hitmap","Occupancy in EndcapHCpID1",208,0.,416.,80,0.,80.);
    Endcap_HCpID1_hitmap->setAxisTitle("Columns",1);
    Endcap_HCpID1_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("HalfCylinder_pI/Disk_2")!=std::string::npos){
    Endcap_HCpID2_ndigis = theDMBE->book1D("Endcap_HCpID2_ndigis","Number of Digis in EndcapHCpID2",50,0.,50.);
    Endcap_HCpID2_ndigis->setAxisTitle("Number of digis",1);
    Endcap_HCpID2_adc = theDMBE->book1D("Endcap_HCpID2_adc","Digi charge in EndcapHCpID2",500,0.,500.);
    Endcap_HCpID2_adc->setAxisTitle("ADC counts",1);
    Endcap_HCpID2_hitmap = theDMBE->book2D("Endcap_HCpID2_hitmap","Occupancy in EndcapHCpID2",208,0.,416.,80,0.,80.);
    Endcap_HCpID2_hitmap->setAxisTitle("Columns",1);
    Endcap_HCpID2_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("HalfCylinder_pO/Disk_1")!=std::string::npos){
    Endcap_HCpOD1_ndigis = theDMBE->book1D("Endcap_SpOD1_ndigis","Number of Digis in EndcapSpOD1",50,0.,50.);
    Endcap_HCpOD1_ndigis->setAxisTitle("Number of digis",1);
    Endcap_HCpOD1_adc = theDMBE->book1D("Endcap_SpOD1_adc","Digi charge in EndcapSpOD1",500,0.,500.);
    Endcap_HCpOD1_adc->setAxisTitle("ADC counts",1);
    Endcap_HCpOD1_hitmap = theDMBE->book2D("Endcap_SpOD1_hitmap","Occupancy in EndcapSpOD1",208,0.,416.,80,0.,80.);
    Endcap_HCpOD1_hitmap->setAxisTitle("Columns",1);
    Endcap_HCpOD1_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else if(currDir.find("HalfCylinder_pO/Disk_2")!=std::string::npos){
    Endcap_HCpOD2_ndigis = theDMBE->book1D("Endcap_SpOD2_ndigis","Number of Digis in EndcapSpOD2",50,0.,50.);
    Endcap_HCpOD2_ndigis->setAxisTitle("Number of digis",1);
    Endcap_HCpOD2_adc = theDMBE->book1D("Endcap_SpOD2_adc","Digi charge in EndcapSpOD2",500,0.,500.);
    Endcap_HCpOD2_adc->setAxisTitle("ADC counts",1);
    Endcap_HCpOD2_hitmap = theDMBE->book2D("Endcap_SpOD2_hitmap","Occupancy in EndcapSpOD2",208,0.,416.,80,0.,80.);
    Endcap_HCpOD2_hitmap->setAxisTitle("Columns",1);
    Endcap_HCpOD2_hitmap->setAxisTitle("Rows",2);
    theDMBE->goUp();
  }else{
    std::vector<std::string> subdirs = theDMBE->getSubdirs();
    for(std::vector<std::string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++){
      if((theDMBE->pwd()).find("Barrel")!=std::string::npos) theDMBE->goUp();
      theDMBE->cd(*it);
      if((theDMBE->pwd()).find("Barrel")!=std::string::npos) continue;
      bookUpperLevelEndcapMEs(theDMBE);
      theDMBE->goUp();
    }
  }

}
//
// Fill histograms
//
void SiPixelDigiModule::fill(bool modon, 
                             const edm::DetSetVector<PixelDigi>& input) {
  
  // Get DQM interface
  //DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
  //std::cout<<"id_ = "<<id_<<" , dmbe="<<theDMBE->pwd()<<std::endl;
  bool BarrelSmIL1=false, BarrelSmIL2=false, BarrelSmIL3=false;
  bool BarrelSmOL1=false, BarrelSmOL2=false, BarrelSmOL3=false;
  bool BarrelSpIL1=false, BarrelSpIL2=false, BarrelSpIL3=false;
  bool BarrelSpOL1=false, BarrelSpOL2=false, BarrelSpOL3=false;
  PXBDetId* pxbdet = new PXBDetId(id_);
  bool EndcapHCmID1=false, EndcapHCmID2=false;
  bool EndcapHCmOD1=false, EndcapHCmOD2=false;
  bool EndcapHCpID1=false, EndcapHCpID2=false;
  bool EndcapHCpOD1=false, EndcapHCpOD2=false;
  //std::cout<<"********************"<<std::endl;
  PXFDetId* pxfdet = new PXFDetId(id_);
  //DetId* det = new DetId(id_);
  //PixelEndcapName* pfn = new PixelEndcapName(*det);
  //PixelBarrelName* pbn = new PixelBarrelName(*det);
  //std::cout<<"pfname: "<<pfn->name()<<", pbname: "<<pbn->name()<<std::endl;
  //std::cout<<"side: "<<pxfdet->side()<<", disk: "<<pxfdet->disk()<<", blade: "<<pxfdet->blade()<<", panel: "<<pxfdet->panel()<<", mod: "<<pxfdet->module()<<std::endl;
  //std::cout<<"layer: "<<pxbdet->layer()<<", ladder: "<<pxbdet->ladder()<<", module: "<<pxbdet->module()<<std::endl;
  if(pxfdet->side()==0){ // barrel!
    if(pxbdet->module()>=1 && pxbdet->module()<=4){ // minus z side
      if(pxbdet->layer()==1){ //Layer1
        if((pxbdet->ladder()>=1 && pxbdet->ladder()<=5)||
	   (pxbdet->ladder()>=16 && pxbdet->ladder()<=20)) BarrelSmIL1 = true;
        else if(pxbdet->ladder()>=6 && pxbdet->ladder()<=15) BarrelSmOL1 = true;
      }else if(pxbdet->layer()==2){ //Layer2
        if((pxbdet->ladder()>=1 && pxbdet->ladder()<=8)||
	   (pxbdet->ladder()>=25 && pxbdet->ladder()<=32)) BarrelSmIL2 = true;
        else if(pxbdet->ladder()>=9 && pxbdet->ladder()<=24) BarrelSmOL2 = true;
      }else if(pxbdet->layer()==3){ //Layer3
        if((pxbdet->ladder()>=1 && pxbdet->ladder()<=11)||
	   (pxbdet->ladder()>=34 && pxbdet->ladder()<=44)) BarrelSmIL3 = true;
        else if(pxbdet->ladder()>=12 && pxbdet->ladder()<=33) BarrelSmOL3 = true;
      }
    }else if(pxbdet->module()>=5 && pxbdet->module()<=8){ // plus z side
      if(pxbdet->layer()==1){ //Layer1
        if((pxbdet->ladder()>=1 && pxbdet->ladder()<=5)||
	   (pxbdet->ladder()>=16 && pxbdet->ladder()<=20)) BarrelSpIL1 = true;
        else if(pxbdet->ladder()>=6 && pxbdet->ladder()<=15) BarrelSpOL1 = true;
      }else if(pxbdet->layer()==2){ //Layer2
        if((pxbdet->ladder()>=1 && pxbdet->ladder()<=8)||
	   (pxbdet->ladder()>=25 && pxbdet->ladder()<=32)) BarrelSpIL2 = true;
        else if(pxbdet->ladder()>=9 && pxbdet->ladder()<=24) BarrelSpOL2 = true;
      }else if(pxbdet->layer()==3){ //Layer3
        if((pxbdet->ladder()>=1 && pxbdet->ladder()<=11)||
	   (pxbdet->ladder()>=34 && pxbdet->ladder()<=44)) BarrelSpIL3 = true;
        else if(pxbdet->ladder()>=12 && pxbdet->ladder()<=33) BarrelSpOL3 = true;
      }
    }
  }if(pxfdet->side()==1){ // minus z side
    if(pxfdet->disk()==1){
      if((pxfdet->blade()>=1 && pxfdet->blade()<=6) ||
         (pxfdet->blade()>=19 && pxfdet->blade()<=24)) EndcapHCmID1 = true;
      else if(pxfdet->blade()>=7 && pxfdet->blade()<=18) EndcapHCmOD1 = true;
    }else if(pxfdet->disk()==2){
      if((pxfdet->blade()>=1 && pxfdet->blade()<=6) ||
         (pxfdet->blade()>=19 && pxfdet->blade()<=24)) EndcapHCmID2 = true;
      else if(pxfdet->blade()>=7 && pxfdet->blade()<=18) EndcapHCmOD2 = true;
    }
  }else if(pxfdet->side()==2){ // plus z side
    if(pxfdet->disk()==1){
      if((pxfdet->blade()>=1 && pxfdet->blade()<=6) ||
         (pxfdet->blade()>=19 && pxfdet->blade()<=24)) EndcapHCpID1 = true;
      else if(pxfdet->blade()>=7 && pxfdet->blade()<=18) EndcapHCpOD1 = true;
    }else if(pxfdet->disk()==2){
      if((pxfdet->blade()>=1 && pxfdet->blade()<=6) ||
         (pxfdet->blade()>=19 && pxfdet->blade()<=24)) EndcapHCpID2 = true;
      else if(pxfdet->blade()>=7 && pxfdet->blade()<=18) EndcapHCpOD2 = true;
    }
  }
  
  
  
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
      if(modon) (mePixDigis_)->Fill((float)col,(float)row);
      if(modon) (meADC_)->Fill((float)adc);
      if(EndcapHCmID1){ Endcap_HCmID1_hitmap->Fill((float)col,(float)row); Endcap_HCmID1_adc->Fill((float)adc);}
      if(EndcapHCmID2){ Endcap_HCmID2_hitmap->Fill((float)col,(float)row); Endcap_HCmID2_adc->Fill((float)adc);}
      if(EndcapHCmOD1){ Endcap_HCmOD1_hitmap->Fill((float)col,(float)row); Endcap_HCmOD1_adc->Fill((float)adc);}
      if(EndcapHCmOD2){ Endcap_HCmOD2_hitmap->Fill((float)col,(float)row); Endcap_HCmOD2_adc->Fill((float)adc);}
      if(EndcapHCpID1){ Endcap_HCpID1_hitmap->Fill((float)col,(float)row); Endcap_HCpID1_adc->Fill((float)adc);}
      if(EndcapHCpID2){ Endcap_HCpID2_hitmap->Fill((float)col,(float)row); Endcap_HCpID2_adc->Fill((float)adc);}
      if(EndcapHCpOD1){ Endcap_HCpOD1_hitmap->Fill((float)col,(float)row); Endcap_HCpOD1_adc->Fill((float)adc);}
      if(EndcapHCpOD2){ Endcap_HCpOD2_hitmap->Fill((float)col,(float)row); Endcap_HCpOD2_adc->Fill((float)adc);}
      if(BarrelSmIL1){ Barrel_SmIL1_hitmap->Fill((float)col,(float)row); Barrel_SmIL1_adc->Fill((float)adc);}
      if(BarrelSmIL2){ Barrel_SmIL2_hitmap->Fill((float)col,(float)row); Barrel_SmIL2_adc->Fill((float)adc);}
      if(BarrelSmIL3){ Barrel_SmIL3_hitmap->Fill((float)col,(float)row); Barrel_SmIL3_adc->Fill((float)adc);}
      if(BarrelSmOL1){ Barrel_SmOL1_hitmap->Fill((float)col,(float)row); Barrel_SmOL1_adc->Fill((float)adc);}
      if(BarrelSmOL2){ Barrel_SmOL2_hitmap->Fill((float)col,(float)row); Barrel_SmOL2_adc->Fill((float)adc);}
      if(BarrelSmOL3){ Barrel_SmOL3_hitmap->Fill((float)col,(float)row); Barrel_SmOL3_adc->Fill((float)adc);}
      if(BarrelSpIL1){ Barrel_SpIL1_hitmap->Fill((float)col,(float)row); Barrel_SpIL1_adc->Fill((float)adc);}
      if(BarrelSpIL2){ Barrel_SpIL2_hitmap->Fill((float)col,(float)row); Barrel_SpIL2_adc->Fill((float)adc);}
      if(BarrelSpIL3){ Barrel_SpIL3_hitmap->Fill((float)col,(float)row); Barrel_SpIL3_adc->Fill((float)adc);}
      if(BarrelSpOL1){ Barrel_SpOL1_hitmap->Fill((float)col,(float)row); Barrel_SpOL1_adc->Fill((float)adc);}
      if(BarrelSpOL2){ Barrel_SpOL2_hitmap->Fill((float)col,(float)row); Barrel_SpOL2_adc->Fill((float)adc);}
      if(BarrelSpOL3){ Barrel_SpOL3_hitmap->Fill((float)col,(float)row); Barrel_SpOL3_adc->Fill((float)adc);}
    }
    if(modon) (meNDigis_)->Fill((float)numberOfDigis);
    if(EndcapHCmID1) Endcap_HCmID1_ndigis->Fill((float)numberOfDigis);
    if(EndcapHCmID2) Endcap_HCmID2_ndigis->Fill((float)numberOfDigis);
    if(EndcapHCmOD1) Endcap_HCmOD1_ndigis->Fill((float)numberOfDigis);
    if(EndcapHCmOD2) Endcap_HCmOD2_ndigis->Fill((float)numberOfDigis);
    if(EndcapHCpID1) Endcap_HCpID1_ndigis->Fill((float)numberOfDigis);
    if(EndcapHCpID2) Endcap_HCpID2_ndigis->Fill((float)numberOfDigis);
    if(EndcapHCpOD1) Endcap_HCpOD1_ndigis->Fill((float)numberOfDigis);
    if(EndcapHCpOD2) Endcap_HCpOD2_ndigis->Fill((float)numberOfDigis);
    if(BarrelSmIL1) Barrel_SmIL1_ndigis->Fill((float)numberOfDigis);
    if(BarrelSmIL2) Barrel_SmIL2_ndigis->Fill((float)numberOfDigis);
    if(BarrelSmIL3) Barrel_SmIL3_ndigis->Fill((float)numberOfDigis);
    if(BarrelSmOL1) Barrel_SmOL1_ndigis->Fill((float)numberOfDigis);
    if(BarrelSmOL2) Barrel_SmOL2_ndigis->Fill((float)numberOfDigis);
    if(BarrelSmOL3) Barrel_SmOL3_ndigis->Fill((float)numberOfDigis);
    if(BarrelSpIL1) Barrel_SpIL1_ndigis->Fill((float)numberOfDigis);
    if(BarrelSpIL2) Barrel_SpIL2_ndigis->Fill((float)numberOfDigis);
    if(BarrelSpIL3) Barrel_SpIL3_ndigis->Fill((float)numberOfDigis);
    if(BarrelSpOL1) Barrel_SpOL1_ndigis->Fill((float)numberOfDigis);
    if(BarrelSpOL2) Barrel_SpOL2_ndigis->Fill((float)numberOfDigis);
    if(BarrelSpOL3) Barrel_SpOL3_ndigis->Fill((float)numberOfDigis);
    //std::cout<<"number of digis="<<numberOfDigis<<std::endl;
      
  }
  
  
  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;
  
}
