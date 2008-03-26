#include "DQM/SiPixelMonitorRecHit/interface/SiPixelRecHitModule.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
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
//
// Fill histograms
//
void SiPixelRecHitModule::fill(const float& rechit_x, const float& rechit_y, const int& sizeX, const int& sizeY) {
  
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
  meXYPos_->Fill(rechit_x, rechit_y);
  meClustX_->Fill(sizeX);
  meClustY_->Fill(sizeY);
  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;
  
}

void SiPixelRecHitModule::nfill(const int& nrec) {

	menRecHits_->Fill(nrec);
}
