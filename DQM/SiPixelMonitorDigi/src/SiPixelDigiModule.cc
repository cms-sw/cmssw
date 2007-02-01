#include "DQM/SiPixelMonitorDigi/interface/SiPixelDigiModule.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

// Framework
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <boost/cstdint.hpp>
#include <string>
#include <stdlib.h>
//
// Constructors
//
SiPixelDigiModule::SiPixelDigiModule() {

}

SiPixelDigiModule::SiPixelDigiModule(uint32_t id): id_(id) { }

//
// Destructor
//
SiPixelDigiModule::~SiPixelDigiModule() {}

//
// Book histograms
//
void SiPixelDigiModule::book() {
  DaqMonitorBEInterface* theDMBE = edm::Service<DaqMonitorBEInterface>().operator->();
  char hkey[80];  
  sprintf(hkey, "ndigis_module_%i",id_);
  meNDigis_ = theDMBE->book1D(hkey,"Number of Digis",50,0.,50.);
  sprintf(hkey, "adc_module_%i",id_);
  meADC_ = theDMBE->book1D(hkey,"Digi charge",500,0.,500.);
  sprintf(hkey, "col_module_%i",id_);
  meCol_ = theDMBE->book1D(hkey,"Digi column",500,0.,500.);
  sprintf(hkey, "row_module_%i",id_);
  meRow_ = theDMBE->book1D(hkey,"Digi row",200,0.,200.);
  sprintf(hkey, "pixdigis_module_%i",id_);
  mePixDigis_ = theDMBE->book2D(hkey,"Digis per four pixels",208,0.,416.,80,0.,160.);
}

//
// Fill histograms
//
void SiPixelDigiModule::fill(const edm::DetSetVector<PixelDigi>& input) {
  
  edm::DetSetVector<PixelDigi>::const_iterator isearch = input.find(id_); // search  digis of detid
  
  if( isearch != input.end() ) {  // Not at empty iterator
    
    unsigned int numberOfDigis = 0;
    
    // Look at digis now
    edm::DetSet<PixelDigi>::const_iterator  di;
    //figure out the size of the module/plaquette:
/*    int maxcol=0, maxrow=0;
    for(di = isearch->data.begin(); di != isearch->data.end(); di++) {
      int col = di->column(); // column 
      int row = di->row();    // row
      if(col>maxcol) maxcol=col;
      if(row>maxrow) maxrow=row;
    }*/
    for(di = isearch->data.begin(); di != isearch->data.end(); di++) {
      numberOfDigis++;
      int adc = di->adc();    // charge
      int col = di->column(); // column 
      int row = di->row();    // row
      (mePixDigis_)->Fill((float)col,(float)row);
      (meADC_)->Fill((float)adc);
      (meCol_)->Fill((float)col);
      (meRow_)->Fill((float)row);
      /*if(subid==2&&adc>0){
	std::cout<<"Plaquette:"<<side<<" , "<<disk<<" , "<<blade<<" , "
	<<panel<<" , "<<zindex<<" ADC="<<adc<<" , COL="<<col<<" , ROW="<<row<<std::endl;
	}else if(subid==1&&adc>0){
	std::cout<<"Module:"<<layer<<" , "<<ladder<<" , "<<zindex<<" ADC="
	<<adc<<" , COL="<<col<<" , ROW="<<row<<std::endl;
	}*/
    }
    (meNDigis_)->Fill((float)numberOfDigis);
    //std::cout<<"number of digis="<<numberOfDigis<<std::endl;
      
  }
  
  
  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;
  
}
