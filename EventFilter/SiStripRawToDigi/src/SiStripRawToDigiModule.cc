#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiModule.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiUnpacker.h"
// 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// 
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigis.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
//
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
//
#include "boost/cstdint.hpp"
#include <cstdlib>

using namespace std;

// -----------------------------------------------------------------------------
//
SiStripRawToDigiModule::SiStripRawToDigiModule( const edm::ParameterSet& pset ) :
  rawToDigi_(0),
  createDigis_( pset.getUntrackedParameter<bool>("CreateDigis",true) )
{
  edm::LogVerbatim("RawToDigi") << "[SiStripRawToDigiModule::SiStripRawToDigiModule] Constructing object...";
  
  int16_t appended_bytes = pset.getUntrackedParameter<int>("AppendedBytes",0);
  int16_t dump_frequency = pset.getUntrackedParameter<int>("FedBufferDumpFreq",0);
  int16_t trigger_fed_id = pset.getUntrackedParameter<int>("TriggerFedId",0);
  bool    using_fed_key  = pset.getUntrackedParameter<bool>("UseFedKey",false);
  rawToDigi_ = new SiStripRawToDigiUnpacker( appended_bytes, 
					     dump_frequency,
					     trigger_fed_id,
					     using_fed_key );
  
  produces< edm::DetSetVector<SiStripRawDigi> >("ScopeMode");
  produces< edm::DetSetVector<SiStripRawDigi> >("VirginRaw");
  produces< edm::DetSetVector<SiStripRawDigi> >("ProcessedRaw");
  produces< edm::DetSetVector<SiStripDigi> >("ZeroSuppressed");
  produces< SiStripDigis >("SiStripDigis");
  produces< SiStripEventSummary >();
  
}

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigiModule::~SiStripRawToDigiModule() {
  edm::LogInfo("RawToDigi") << "[SiStripRawToDigiModule::~SiStripRawToDigiModule] Destructing object...";
  if ( rawToDigi_ ) delete rawToDigi_;
}

// -----------------------------------------------------------------------------
/** 
    Retrieves cabling map from EventSetup, retrieves
    FEDRawDataCollection from Event, creates a DetSetVector of
    SiStripDigis (EDProduct), uses RawToDigi converter to fill the
    DetSetVector, attaches StripDigiCollection to Event.
*/
void SiStripRawToDigiModule::produce( edm::Event& iEvent, 
				      const edm::EventSetup& iSetup ) {
  
  // Retrieve FED cabling
  edm::ESHandle<SiStripFedCabling> cabling;
  iSetup.get<SiStripFedCablingRcd>().get( cabling );

  // Retrieve FED raw data ("source" label is now fixed by fwk)
  edm::Handle<FEDRawDataCollection> buffers;
  iEvent.getByLabel( "source", buffers ); 
  
  // Create auto pointers for products
  auto_ptr< edm::DetSetVector<SiStripRawDigi> > sm( new edm::DetSetVector<SiStripRawDigi> );
  auto_ptr< edm::DetSetVector<SiStripRawDigi> > vr( new edm::DetSetVector<SiStripRawDigi> );
  auto_ptr< edm::DetSetVector<SiStripRawDigi> > pr( new edm::DetSetVector<SiStripRawDigi> );
  auto_ptr< edm::DetSetVector<SiStripDigi> > zs( new edm::DetSetVector<SiStripDigi> );
  auto_ptr<SiStripEventSummary> summary( new SiStripEventSummary() );
  auto_ptr<SiStripDigis> digis( new SiStripDigis() );

  // Create "real" or "pseudo" digis
  if ( !createDigis_ ) { rawToDigi_->createDigis( cabling, buffers, digis ); }
  else { rawToDigi_->createDigis( cabling, buffers, sm, vr, pr, zs ); }
  
  // Populate SiStripEventSummary object with "trigger FED" info
  rawToDigi_->triggerFed( buffers, summary ); 
  
  iEvent.put( sm, "ScopeMode" );
  iEvent.put( vr, "VirginRaw" );
  iEvent.put( pr, "ProcessedRaw" );
  iEvent.put( zs, "ZeroSuppressed" );
  iEvent.put( digis, "SiStripDigis" );
  iEvent.put( summary );
  
}

