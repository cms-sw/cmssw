#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiModule.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigi.h"
// edm
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// data formats
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigis.h"
// cabling
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
// std
#include <cstdlib>

// -----------------------------------------------------------------------------
/** 
    Creates instance of RawToDigi converter, defines EDProduct type.
*/
SiStripRawToDigiModule::SiStripRawToDigiModule( const edm::ParameterSet& pset ) :
  inputModuleLabel_( pset.getParameter<string>( "InputModuleLabel" ) ),
  rawToDigi_(0)
{
  edm::LogInfo("RawToDigi") << "[SiStripRawToDigiModule::SiStripRawToDigiModule] Constructing object...";
  int16_t bytes  = pset.getUntrackedParameter<int>("AppendedBytes",0);
  int16_t freq   = pset.getUntrackedParameter<int>("FedBufferDumpFreq",0);
  bool    fedkey = pset.getUntrackedParameter<bool>("UseFedKey",false);
  int16_t fedid  = pset.getUntrackedParameter<int>("TriggerFedId",1023);
  rawToDigi_ = new SiStripRawToDigi( bytes, freq, fedkey, fedid );
  
  produces< edm::DetSetVector<SiStripRawDigi> >("ScopeMode");
  produces< edm::DetSetVector<SiStripRawDigi> >("VirginRaw");
  produces< edm::DetSetVector<SiStripRawDigi> >("ProcessedRaw");
  produces< edm::DetSetVector<SiStripDigi> >   ("ZeroSuppressed");
  produces<SiStripEventSummary>();
  produces<SiStripDigi>();
  //produces<SiStripDigi>("SiStripDigis");
  
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
  
  if ( !(iEvent.id().event()%10) ) {
    edm::LogInfo("RawToDigi") << "[SiStripRawToDigiModule::produce]"
			      << " Processing event number: " << iEvent.id().event();
  }
  
  edm::ESHandle<SiStripFedCabling> cabling;
  iSetup.get<SiStripFedCablingRcd>().get( cabling );
  
  edm::Handle<FEDRawDataCollection> buffers;
  iEvent.getByLabel( inputModuleLabel_, buffers );
  
  auto_ptr< edm::DetSetVector<SiStripRawDigi> > sm( new edm::DetSetVector<SiStripRawDigi> );
  auto_ptr< edm::DetSetVector<SiStripRawDigi> > vr( new edm::DetSetVector<SiStripRawDigi> );
  auto_ptr< edm::DetSetVector<SiStripRawDigi> > pr( new edm::DetSetVector<SiStripRawDigi> );
  auto_ptr< edm::DetSetVector<SiStripDigi> > zs( new edm::DetSetVector<SiStripDigi> );
  auto_ptr<SiStripEventSummary> summary( new SiStripEventSummary() );
  auto_ptr<SiStripDigis> digis;
  
  rawToDigi_->createDigis( iEvent.id().event(), cabling, buffers, sm, vr, pr, zs, summary, digis );
  
  iEvent.put( sm, "ScopeMode" );
  iEvent.put( vr, "VirginRaw" );
  iEvent.put( pr, "ProcessedRaw" );
  iEvent.put( zs, "ZeroSuppressed" );
  iEvent.put( summary );
  //iEvent.put( digis, "SiStripDigis" );
  iEvent.put( digis );
  
}

