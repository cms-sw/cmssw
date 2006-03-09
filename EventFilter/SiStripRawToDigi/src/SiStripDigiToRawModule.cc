#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRawModule.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h"
// edm
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
// data formats
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
// cabling
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
// std
#include <cstdlib>

// -----------------------------------------------------------------------------
/** 
    Creates instance of DigiToRaw converter, defines EDProduct type.
*/
SiStripDigiToRawModule::SiStripDigiToRawModule( const edm::ParameterSet& pset ) :
  digiToRaw_(0),
  eventCounter_(0)
{
  cout << "[SiStripDigiToRawModule::SiStripDigiToRawModule]"
       << " Constructing object..." << endl;
  
  // Create instance of DigiToRaw formatter
  string mode    = pset.getUntrackedParameter<std::string>("fedReadoutMode","VIRGIN_RAW");
  int16_t nbytes = pset.getUntrackedParameter<int>("nAppendedBytes",0);
  digiToRaw_ = new SiStripDigiToRaw( mode, nbytes );
  
  produces<FEDRawDataCollection>("FromSimulation");

}

// -----------------------------------------------------------------------------
/** */
SiStripDigiToRawModule::~SiStripDigiToRawModule() {
  cout << "[SiStripDigiToRawModule::~SiStripDigiToRawModule]"
       << " Destructing object..." << endl;
  if ( digiToRaw_ ) delete digiToRaw_;
}

// -----------------------------------------------------------------------------
/** 
    Retrieves cabling map from EventSetup, retrieves a DetSetVector of
    SiStirpDigis from Event, creates a FEDRawDataCollection
    (EDProduct), uses DigiToRaw converter to fill
    FEDRawDataCollection, attaches FEDRawDataCollection to Event.
*/
void SiStripDigiToRawModule::produce( edm::Event& iEvent, 
				      const edm::EventSetup& iSetup ) {

  eventCounter_++; 
  cout << "[SiStripDigiToRawModule::produce] "
       << "event number: " << eventCounter_ << endl;
  
  edm::ESHandle<SiStripFedCabling> cabling;
  iSetup.get<SiStripFedCablingRcd>().get( cabling );
  
  edm::Handle< edm::DetSetVector<SiStripDigi> > digis;
  iEvent.getByType( digis );
  
  std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );
  
  digiToRaw_->createFedBuffers( cabling, digis, buffers );

  iEvent.put( buffers );

}
