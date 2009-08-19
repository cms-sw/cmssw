// Last commit: $Id: SiStripDigiToRawModule.cc,v 1.8 2009/03/27 10:22:16 bainbrid Exp $

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
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include <cstdlib>

namespace sistrip {

  // -----------------------------------------------------------------------------
  /** 
      Creates instance of DigiToRaw converter, defines EDProduct type.
  */
  DigiToRawModule::DigiToRawModule( const edm::ParameterSet& pset ) :
    inputModuleLabel_( pset.getParameter<std::string>( "InputModuleLabel" ) ),
    inputDigiLabel_( pset.getParameter<std::string>( "InputDigiLabel" ) ),
    mode_( fedReadoutModeFromString(pset.getParameter<std::string>( "FedReadoutMode" ))),
    rawdigi_( false ),
    digiToRaw_(0),
    eventCounter_(0)
  {
    if ( edm::isDebugEnabled() ) {
      LogDebug("DigiToRawModule") 
	<< "[sistrip::DigiToRawModule::DigiToRawModule]"
	<< " Constructing object...";
    }  
    
    switch(mode_) {
    case READOUT_MODE_ZERO_SUPPRESSED_LITE: rawdigi_ = false; break;
    case READOUT_MODE_ZERO_SUPPRESSED:      rawdigi_ = false; break;
    case READOUT_MODE_VIRGIN_RAW:      rawdigi_ = true; break;
    case READOUT_MODE_PROC_RAW:        rawdigi_ = true; break;
    case READOUT_MODE_SCOPE:           rawdigi_ = true; break;
    case READOUT_MODE_INVALID: {
      if( edm::isDebugEnabled()) {
	edm::LogWarning("DigiToRawModule") 
	  << "[sistrip::DigiToRawModule::DigiToRawModule]"
	  << " UNKNOWN readout mode: " << pset.getParameter<std::string>("FedReadoutMode");
      }} break;
    }
    if(pset.getParameter<bool>("UseWrongDigiType")) {
      rawdigi_ = !rawdigi_;
      if( edm::isDebugEnabled()) {
	edm::LogWarning("DigiToRawModule") 
	  << "[sistrip::DigiToRawModule::DigiToRawModule]"
	  << " You are using the wrong type of digis!";
      }
    }

    // Create instance of DigiToRaw formatter
    digiToRaw_ = new DigiToRaw( mode_, pset.getParameter<bool>("UseFedKey") );
  
    produces<FEDRawDataCollection>();

  }

  // -----------------------------------------------------------------------------
  /** */
  DigiToRawModule::~DigiToRawModule() {
    if ( edm::isDebugEnabled() ) {
      LogDebug("DigiToRaw")
	<< "[sistrip::DigiToRawModule::~DigiToRawModule]"
	<< " Destructing object...";
    }
    if ( digiToRaw_ ) delete digiToRaw_;
  }

  // -----------------------------------------------------------------------------
  /** 
      Retrieves cabling map from EventSetup, retrieves a DetSetVector of
      SiStripDigis from Event, creates a FEDRawDataCollection
      (EDProduct), uses DigiToRaw converter to fill
      FEDRawDataCollection, attaches FEDRawDataCollection to Event.
  */
  void DigiToRawModule::produce( edm::Event& iEvent, 
				 const edm::EventSetup& iSetup ) {

    eventCounter_++; 
  
    std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );

    edm::ESHandle<SiStripFedCabling> cabling;
    iSetup.get<SiStripFedCablingRcd>().get( cabling );

    if( rawdigi_ ) {
      edm::Handle< edm::DetSetVector<SiStripRawDigi> > rawdigis;
      iEvent.getByLabel( inputModuleLabel_, inputDigiLabel_, rawdigis );
      digiToRaw_->createFedBuffers( iEvent, cabling, rawdigis, buffers );
    } else {
      edm::Handle< edm::DetSetVector<SiStripDigi> > digis;
      iEvent.getByLabel( inputModuleLabel_, inputDigiLabel_, digis );
      digiToRaw_->createFedBuffers( iEvent, cabling, digis, buffers );
    }

    iEvent.put( buffers );
  
  }

}


////////////////////////////////////////////////////////////////////////////////
//@@@ TO BE DEPRECATED BELOW!!!


// -----------------------------------------------------------------------------
/** 
    Creates instance of DigiToRaw converter, defines EDProduct type.
*/
OldSiStripDigiToRawModule::OldSiStripDigiToRawModule( const edm::ParameterSet& pset ) :
  inputModuleLabel_( pset.getParameter<std::string>( "InputModuleLabel" ) ),
  inputDigiLabel_( pset.getParameter<std::string>( "InputDigiLabel" ) ),
  digiToRaw_(0),
  eventCounter_(0)
{
  if ( edm::isDebugEnabled() ) {
    LogDebug("DigiToRaw") 
      << "[OldSiStripDigiToRawModule::OldSiStripDigiToRawModule]"
      << " Constructing object...";
  }

  // Create instance of DigiToRaw formatter
  std::string mode = pset.getUntrackedParameter<std::string>("FedReadoutMode","VIRGIN_RAW");
  int16_t nbytes = pset.getUntrackedParameter<int>("AppendedBytes",0);
  bool use_fed_key = pset.getUntrackedParameter<bool>("UseFedKey",false);
  digiToRaw_ = new OldSiStripDigiToRaw( mode, nbytes, use_fed_key );
  
  produces<FEDRawDataCollection>();

}

// -----------------------------------------------------------------------------
/** */
OldSiStripDigiToRawModule::~OldSiStripDigiToRawModule() {
  if ( edm::isDebugEnabled() ) {
    LogDebug("DigiToRaw")
      << "[OldSiStripDigiToRawModule::~OldSiStripDigiToRawModule]"
      << " Destructing object...";
  }
  if ( digiToRaw_ ) delete digiToRaw_;
}

// -----------------------------------------------------------------------------
/** 
    Retrieves cabling map from EventSetup, retrieves a DetSetVector of
    SiStripDigis from Event, creates a FEDRawDataCollection
    (EDProduct), uses DigiToRaw converter to fill
    FEDRawDataCollection, attaches FEDRawDataCollection to Event.
*/
void OldSiStripDigiToRawModule::produce( edm::Event& iEvent, 
				      const edm::EventSetup& iSetup ) {

  eventCounter_++; 
  
  edm::ESHandle<SiStripFedCabling> cabling;
  iSetup.get<SiStripFedCablingRcd>().get( cabling );

  edm::Handle< edm::DetSetVector<SiStripDigi> > digis;
  iEvent.getByLabel( inputModuleLabel_, inputDigiLabel_, digis );

  std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );
  
  digiToRaw_->createFedBuffers( iEvent, cabling, digis, buffers );

  iEvent.put( buffers );
  
}
