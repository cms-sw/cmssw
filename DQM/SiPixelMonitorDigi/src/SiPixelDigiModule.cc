#include "DQM/SiPixelMonitorDigi/interface/SiPixelDigiModule.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
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
// Book
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
}

//
// Book
//
void SiPixelDigiModule::fill(const PixelDigiCollection* digiCollection) {

  const std::vector<unsigned int> detIDs = digiCollection->detIDs();
 
  std::vector<unsigned int>::const_iterator detunit_it  = detIDs.begin(), detunit_end = detIDs.end();
  int numberOfDigis = 0;
  
  for ( ; detunit_it != detunit_end; ++detunit_it ) {
    if( id_ ==(*detunit_it)) {
      const PixelDigiCollection::Range digiRange = digiCollection->get(id_);
      
      PixelDigiCollection::ContainerIterator digiBegin = digiRange.first;
      PixelDigiCollection::ContainerIterator digiEnd   = digiRange.second;
      PixelDigiCollection::ContainerIterator di = digiBegin;
      //std::cout << " *** SiPixelDigiModule::fill - Filling Detid " << id_ << " size " << digiEnd - digiBegin << std::endl;
 
      for( ; di != digiEnd; ++di) {
	numberOfDigis++;
	int adc = di->adc();
	// std::cout << " adc " << adc << std::endl;
	int col = di->column();
	int row = di->row();
	(meADC_)->Fill((float)adc);
	(meCol_)->Fill((float)col);
	(meRow_)->Fill((float)row);
	//       //cout << " DetID: " << detid << " Col: " << col << " Row: " << row << " ADC: " << adc << endl;
      }
    }
  }
 (meNDigis_)->Fill((float)numberOfDigis); 
  
}
