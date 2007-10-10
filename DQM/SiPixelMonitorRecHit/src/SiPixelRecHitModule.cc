#include "DQM/SiPixelMonitorRecHit/interface/SiPixelRecHitModule.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
/// Framework
#include "FWCore/Framework/interface/ESHandle.h"
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
void SiPixelRecHitModule::book(const edm::ParameterSet& iConfig) {

  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = iConfig.getParameter<edm::InputTag>( "src" );
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );
  // Get DQM interface
  DaqMonitorBEInterface* theDMBE = edm::Service<DaqMonitorBEInterface>().operator->();
  // XYPosition
  hid = theHistogramId->setHistoId("xypos",id_);
  //std::cout << hid << " " << theHistogramId->getDataCollection(hid) << " " << theHistogramId->getRawId(hid) << std::endl;
  meXYPos_ = theDMBE->book2D(hid,"XY Position",100,-4.,4,100,-4,4);
  meXYPos_->setAxisTitle("X Position",1);
  meXYPos_->setAxisTitle("Y Position",2);
  // X and Y Resolution
  hid = theHistogramId->setHistoId("Xres",id_);
  meXRes_ = theDMBE->book1D(hid,"X Resolution",100,-200,200.);
  meXRes_->setAxisTitle("X Resolution",1);
  hid = theHistogramId->setHistoId("Yres",id_);
  meYRes_ = theDMBE->book1D(hid,"Y Resolution",100,-200,200.);
  meYRes_->setAxisTitle("Y Resolution",1);
  //X and Y Pull
  hid = theHistogramId->setHistoId("Xpull",id_);
  meXPull_ = theDMBE->book1D(hid,"X Pull",100,-200,200.);
  meXPull_->setAxisTitle("X Pull",1);
  hid = theHistogramId->setHistoId("Ypull",id_);
  meYPull_ = theDMBE->book1D(hid,"Y Pull",100,-200,200.);
  meYPull_->setAxisTitle("Y Pull",1);
  delete theHistogramId;
}
//
// Fill histograms
//
void SiPixelRecHitModule::fill(const float& rechit_x, const float& rechit_y, const float& x_res, const float& y_res, const float& x_pull, const float& y_pull) {
  
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
  
  meXYPos_->Fill(rechit_x, rechit_y);
  meXRes_->Fill(x_res);
  meYRes_->Fill(y_res);
  meXPull_->Fill(x_pull);
  meYPull_->Fill(y_pull);

  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;
  
}
