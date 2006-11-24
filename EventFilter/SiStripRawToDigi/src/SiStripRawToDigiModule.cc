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
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigiCollection.h"
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
using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripRawToDigiModule::SiStripRawToDigiModule( const edm::ParameterSet& pset ) :
  rawToDigi_(0),
  createDigis_( pset.getUntrackedParameter<bool>("CreateDigis",true) )
{
  LogTrace(mlRawToDigi_)
    << "[SiStripRawToDigiModule::" << __func__ << "]"
    << " Constructing object...";
  
  int16_t appended_bytes = pset.getUntrackedParameter<int>("AppendedBytes",0);
  int16_t fed_buffer_dump_freq = pset.getUntrackedParameter<int>("FedBufferDumpFreq",0);
  int16_t fed_event_dump_freq = pset.getUntrackedParameter<int>("FedEventDumpFreq",0);
  int16_t trigger_fed_id = pset.getUntrackedParameter<int>("TriggerFedId",0);
  bool    using_fed_key  = pset.getUntrackedParameter<bool>("UseFedKey",false);
  rawToDigi_ = new SiStripRawToDigiUnpacker( appended_bytes, 
					     fed_buffer_dump_freq,
					     fed_event_dump_freq,
					     trigger_fed_id,
					     using_fed_key );
  
  produces< SiStripEventSummary >();
  produces< edm::DetSetVector<SiStripRawDigi> >("ScopeMode");
  produces< edm::DetSetVector<SiStripRawDigi> >("VirginRaw");
  produces< edm::DetSetVector<SiStripRawDigi> >("ProcessedRaw");
  produces< edm::DetSetVector<SiStripDigi> >("ZeroSuppressed");
  produces< SiStripDigiCollection >("SiStripDigiCollection");
  
  createDigis_ = true; //@@ force this for the time being...  

}

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigiModule::~SiStripRawToDigiModule() {
  if ( rawToDigi_ ) delete rawToDigi_;
  LogTrace(mlRawToDigi_)
    << "[SiStripRawToDigiModule::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** 
    Retrieves cabling map from EventSetup, retrieves
    FEDRawDataCollection from Event, creates a DetSetVector of
    SiStripDigiCollection (EDProduct), uses RawToDigi converter to fill the
    DetSetVector, attaches StripDigiCollection to Event.
*/
void SiStripRawToDigiModule::produce( edm::Event& event, 
				      const edm::EventSetup& setup ) {
  
  // Retrieve FED cabling
  edm::ESHandle<SiStripFedCabling> cabling;
  setup.get<SiStripFedCablingRcd>().get( cabling );

  // Retrieve FED raw data ("source" label is now fixed by fwk)
  edm::Handle<FEDRawDataCollection> buffers;
  //event.getByType( buffers ); 
  event.getByLabel( "source", buffers ); 
  
  // Populate SiStripEventSummary object with "trigger FED" info
  auto_ptr<SiStripEventSummary> summary( new SiStripEventSummary() );
  rawToDigi_->triggerFed( *buffers, *summary, event.id().event() ); 
  
  // Create auto pointers for digi products
  auto_ptr< edm::DetSetVector<SiStripRawDigi> > sm( new edm::DetSetVector<SiStripRawDigi> );
  auto_ptr< edm::DetSetVector<SiStripRawDigi> > vr( new edm::DetSetVector<SiStripRawDigi> );
  auto_ptr< edm::DetSetVector<SiStripRawDigi> > pr( new edm::DetSetVector<SiStripRawDigi> );
  auto_ptr< edm::DetSetVector<SiStripDigi> > zs( new edm::DetSetVector<SiStripDigi> );
  auto_ptr<SiStripDigiCollection> digis( new SiStripDigiCollection() );

  // Create "real" or "pseudo" digis
  if ( !createDigis_ ) { rawToDigi_->createDigis( *cabling, buffers, *summary, digis ); }
  else { rawToDigi_->createDigis( *cabling, *buffers, *summary, *sm, *vr, *pr, *zs ); }
  
  event.put( summary );
  event.put( sm, "ScopeMode" );
  event.put( vr, "VirginRaw" );
  event.put( pr, "ProcessedRaw" );
  event.put( zs, "ZeroSuppressed" );
  event.put( digis, "SiStripDigiCollection" );
  
}

