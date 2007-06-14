// Last commit: $Id: SiStripRawToDigiModule.cc,v 1.1 2007/04/24 16:58:58 bainbrid Exp $

#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToDigiModule.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiUnpacker.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "boost/cstdint.hpp"
#include <cstdlib>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripRawToDigiModule::SiStripRawToDigiModule( const edm::ParameterSet& pset ) :
  rawToDigi_(0),
  label_( pset.getUntrackedParameter<std::string>("ProductLabel","source") ),
  instance_( pset.getUntrackedParameter<std::string>("ProductInstance","") )
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
  
}

// -----------------------------------------------------------------------------
/** */
SiStripRawToDigiModule::~SiStripRawToDigiModule() {
  if ( rawToDigi_ ) { delete rawToDigi_; }
  LogTrace(mlRawToDigi_)
    << "[SiStripRawToDigiModule::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** 
    Retrieves cabling map from EventSetup and FEDRawDataCollection
    from Event, creates a DetSetVector of SiStrip(Raw)Digis, uses the
    SiStripRawToDigiUnpacker class to fill the DetSetVector, and
    attaches the container to the Event.
*/
void SiStripRawToDigiModule::produce( edm::Event& event, 
				      const edm::EventSetup& setup ) {

  LogTrace(mlRawToDigi_) 
    << "[SiStripRawToDigiModule::" << __func__ << "]"
    << " Analyzing run/event "
    << event.id().run() << "/"
    << event.id().event();
  
  // Retrieve FED cabling
  edm::ESHandle<SiStripFedCabling> cabling;
  setup.get<SiStripFedCablingRcd>().get( cabling );

  // Retrieve FED raw data (by label, which is "source" by default)
  edm::Handle<FEDRawDataCollection> buffers;
  event.getByLabel( label_, instance_, buffers ); 

  // Populate SiStripEventSummary object with "trigger FED" info
  std::auto_ptr<SiStripEventSummary> summary( new SiStripEventSummary() );
  rawToDigi_->triggerFed( *buffers, *summary, event.id().event() ); 

  // Create containers for digis
  edm::DetSetVector<SiStripRawDigi>* sm = new edm::DetSetVector<SiStripRawDigi>();
  edm::DetSetVector<SiStripRawDigi>* vr = new edm::DetSetVector<SiStripRawDigi>();
  edm::DetSetVector<SiStripRawDigi>* pr = new edm::DetSetVector<SiStripRawDigi>();
  edm::DetSetVector<SiStripDigi>* zs = new edm::DetSetVector<SiStripDigi>();

  // Create digis
  if ( rawToDigi_ ) { rawToDigi_->createDigis( *cabling,*buffers,*summary,*sm,*vr,*pr,*zs ); }
  
  // Create auto_ptr's of digi products
  std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > sm_dsv(sm);
  std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > vr_dsv(vr);
  std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > pr_dsv(pr);
  std::auto_ptr< edm::DetSetVector<SiStripDigi> > zs_dsv(zs);
  
  // Add to event
  event.put( summary );
  event.put( sm_dsv, "ScopeMode" );
  event.put( vr_dsv, "VirginRaw" );
  event.put( pr_dsv, "ProcessedRaw" );
  event.put( zs_dsv, "ZeroSuppressed" );
  
}

