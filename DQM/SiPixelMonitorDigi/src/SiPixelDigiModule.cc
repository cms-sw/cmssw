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
void SiPixelDigiModule::book(const edm::ParameterSet& iConfig) {
  
  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = iConfig.getParameter<edm::InputTag>( "src" );
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );
  // Get DQM interface
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
  
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
  
  delete theHistogramId;
  
}


//
// Fill histograms
//
void SiPixelDigiModule::fill(const edm::DetSetVector<PixelDigi>& input) {
  
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
      (mePixDigis_)->Fill((float)col,(float)row);
      (meADC_)->Fill((float)adc);
    }
    (meNDigis_)->Fill((float)numberOfDigis);
    //std::cout<<"number of digis="<<numberOfDigis<<std::endl;
      
  }
  
  
  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;
  
}
