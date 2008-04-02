// Last commit: $Id: SiStripDigiToRawModule.cc,v 1.4 2008/01/22 19:28:07 muzaffar Exp $

#include "EventFilter/SiStripRawToDigi/plugins/SiStripDigiToRawModule.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include <cstdlib>

using namespace std;

// -----------------------------------------------------------------------------
/** 
    Creates instance of DigiToRaw converter, defines EDProduct type.
*/
SiStripDigiToRawModule::SiStripDigiToRawModule( const edm::ParameterSet& pset ) :
  inputModuleLabel_( pset.getParameter<std::string>( "InputModuleLabel" ) ),
  inputDigiLabel_( pset.getParameter<std::string>( "InputDigiLabel" ) ),
  digiToRaw_(0),
  eventCounter_(0)
{
  LogDebug("DigiToRaw") << "[SiStripDigiToRawModule::SiStripDigiToRawModule] Constructing object...";
  
  // Create instance of DigiToRaw formatter
  std::string mode = pset.getUntrackedParameter<std::string>("FedReadoutMode","VIRGIN_RAW");
  int16_t nbytes = pset.getUntrackedParameter<int>("AppendedBytes",0);
  bool use_fed_key = pset.getUntrackedParameter<bool>("UseFedKey",false);
  digiToRaw_ = new SiStripDigiToRaw( mode, nbytes, use_fed_key );
  
  produces<FEDRawDataCollection>();

}

// -----------------------------------------------------------------------------
/** */
SiStripDigiToRawModule::~SiStripDigiToRawModule() {
  LogDebug("DigiToRaw") << "[SiStripDigiToRawModule::~SiStripDigiToRawModule] Destructing object...";
  if ( digiToRaw_ ) delete digiToRaw_;
}

// -----------------------------------------------------------------------------
/** 
    Retrieves cabling map from EventSetup, retrieves a DetSetVector of
    SiStripDigis from Event, creates a FEDRawDataCollection
    (EDProduct), uses DigiToRaw converter to fill
    FEDRawDataCollection, attaches FEDRawDataCollection to Event.
*/
void SiStripDigiToRawModule::produce( edm::Event& iEvent, 
				      const edm::EventSetup& iSetup ) {

  eventCounter_++; 
  LogDebug("DigiToRaw") << "[SiStripDigiToRawModule::produce] Event number: " << eventCounter_;
  
  edm::ESHandle<SiStripFedCabling> cabling;
  iSetup.get<SiStripFedCablingRcd>().get( cabling );

  edm::Handle< edm::DetSetVector<SiStripDigi> > digis;
  iEvent.getByLabel( inputModuleLabel_, inputDigiLabel_, digis );

  std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );
  
  digiToRaw_->createFedBuffers( iEvent, cabling, digis, buffers );

  iEvent.put( buffers );
  
}
