#include "SiStripDigiToRawModule.h"
#include "SiStripDigiToRaw.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <cstdlib>

namespace sistrip {
	
  //fill Descriptions needed to define default parameters	
  void DigiToRawModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("FedReadoutMode", "ZERO_SUPPRESSED");
    desc.add<std::string>("PacketCode", "ZERO_SUPPRESSED");
    desc.add<bool>("UseFedKey", false);
    desc.add<bool>("UseWrongDigiType", false);
    desc.add<bool>("CopyBufferHeader", false);
    desc.add<edm::InputTag>("InputDigis", edm::InputTag("simSiStripDigis", "ZeroSuppressed"));
    desc.add<edm::InputTag>("RawDataTag", edm::InputTag("rawDataCollector"));
    descriptions.add("SiStripDigiToRawModule",desc);
  }

  // -----------------------------------------------------------------------------
  /** 
      Creates instance of DigiToRaw converter, defines EDProduct type.
  */
  DigiToRawModule::DigiToRawModule( const edm::ParameterSet& pset ) :
    copyBufferHeader_(pset.getParameter<bool>("CopyBufferHeader")),
    mode_( fedReadoutModeFromString(pset.getParameter<std::string>( "FedReadoutMode" ))),
    packetCode_(packetCodeFromString(pset.getParameter<std::string>("PacketCode"), mode_)),
    rawdigi_( false ),
    digiToRaw_(nullptr),
    eventCounter_(0),
    inputDigiTag_(pset.getParameter<edm::InputTag>("InputDigis")),
    rawDataTag_(pset.getParameter<edm::InputTag>("RawDataTag"))
  {
    if ( edm::isDebugEnabled() ) {
      LogDebug("DigiToRawModule") 
	<< "[sistrip::DigiToRawModule::DigiToRawModule]"
	<< " Constructing object...";
    }  

    
    switch(mode_) {
    case READOUT_MODE_ZERO_SUPPRESSED:                         rawdigi_ = false; break;
    case READOUT_MODE_ZERO_SUPPRESSED_FAKE:                    rawdigi_ = false; break;
    case READOUT_MODE_ZERO_SUPPRESSED_LITE10:                  rawdigi_ = false; break;
    case READOUT_MODE_ZERO_SUPPRESSED_LITE10_CMOVERRIDE:       rawdigi_ = false; break;
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8:                   rawdigi_ = false; break;
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_CMOVERRIDE:        rawdigi_ = false; break;
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT:            rawdigi_ = false; break;
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT_CMOVERRIDE: rawdigi_ = false; break;
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT:            rawdigi_ = false; break;
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT_CMOVERRIDE: rawdigi_ = false; break;
    case READOUT_MODE_PREMIX_RAW:                              rawdigi_ = false; break; 
    case READOUT_MODE_VIRGIN_RAW:                              rawdigi_ = true; break;
    case READOUT_MODE_PROC_RAW:                                rawdigi_ = true; break;
    case READOUT_MODE_SCOPE:                                   rawdigi_ = true; break;
    case READOUT_MODE_INVALID: {
      if( edm::isDebugEnabled()) {
	edm::LogWarning("DigiToRawModule") 
	  << "[sistrip::DigiToRawModule::DigiToRawModule]"
	  << " UNKNOWN readout mode: " << pset.getParameter<std::string>("FedReadoutMode");
      }} break;
    case READOUT_MODE_SPY: {
      if( edm::isDebugEnabled()) {
	edm::LogWarning("DigiToRawModule") 
	  << "[sistrip::DigiToRawModule::DigiToRawModule]"
	  << " Digi to raw is not supported for spy channel data";
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
    digiToRaw_ = new DigiToRaw(mode_, packetCode_, pset.getParameter<bool>("UseFedKey"));

    if (rawdigi_) {
      tokenRawDigi = consumes< edm::DetSetVector<SiStripRawDigi> >(inputDigiTag_);
    } else {
      tokenDigi = consumes< edm::DetSetVector<SiStripDigi> >(inputDigiTag_);
    }
    if (copyBufferHeader_){
      //CAMM input raw module label or same as digi ????
      if( edm::isDebugEnabled()) {
	edm::LogWarning("DigiToRawModule") 
	  << "[sistrip::DigiToRawModule::DigiToRawModule]"
	  << "Copying buffer header from collection " << rawDataTag_;
      }
      tokenRawBuffer = consumes<FEDRawDataCollection>(rawDataTag_);
    }

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
  
    auto buffers = std::make_unique<FEDRawDataCollection>();

    edm::ESHandle<SiStripFedCabling> cabling;
    iSetup.get<SiStripFedCablingRcd>().get( cabling );

    //get buffer header from original rawdata
    edm::Handle<FEDRawDataCollection> rawbuffers;
    if (copyBufferHeader_){
      if( edm::isDebugEnabled()) {
	edm::LogWarning("DigiToRawModule") 
	  << "[sistrip::DigiToRawModule::DigiToRawModule]"
	  << "Getting raw buffer: ";
      }
      try {
	iEvent.getByToken( tokenRawBuffer, rawbuffers );
      } catch (const cms::Exception& e){
	if( edm::isDebugEnabled()) {
	  edm::LogWarning("DigiToRawModule") 
	    << "[sistrip::DigiToRawModule::DigiToRawModule]"
	    << " Failed to get collection " << rawDataTag_;
	}
      }
    }

    if( rawdigi_ ) {
      edm::Handle< edm::DetSetVector<SiStripRawDigi> > rawdigis;
      iEvent.getByToken( tokenRawDigi, rawdigis );
      if (copyBufferHeader_) digiToRaw_->createFedBuffers( iEvent, cabling, rawbuffers, rawdigis, buffers );
      else digiToRaw_->createFedBuffers( iEvent, cabling, rawdigis, buffers );
    } else {
      edm::Handle< edm::DetSetVector<SiStripDigi> > digis;
      iEvent.getByToken( tokenDigi, digis );
      if (copyBufferHeader_) digiToRaw_->createFedBuffers( iEvent, cabling, rawbuffers, digis, buffers );
      else digiToRaw_->createFedBuffers( iEvent, cabling, digis, buffers );
    }


    iEvent.put(std::move(buffers));
  
  }

  void DigiToRawModule::endStream()
  {
    digiToRaw_->printWarningSummary();
  }
}

