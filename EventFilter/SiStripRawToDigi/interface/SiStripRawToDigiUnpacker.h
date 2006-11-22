#ifndef EventFilter_SiStripRawToDigi_SiStripRawToDigiUnpacker_H
#define EventFilter_SiStripRawToDigi_SiStripRawToDigiUnpacker_H

#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "boost/cstdint.hpp"
#include <iostream>
#include <string>
#include <vector>
 
namespace Fed9U { class Fed9UEvent; }
class FEDRawDataCollection;
class FEDRawData;
class SiStripDigiCollection;
class SiStripDigi;
class SiStripRawDigi;
class SiStripEventSummary;
class SiStripFedCabling;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiUnpacker.h
   @class SiStripRawToDigiUnpacker 
   @brief Takes collection of FEDRawData as input and creates digis.
*/
class SiStripRawToDigiUnpacker {
  
 public:
  
  /** */
  SiStripRawToDigiUnpacker( int16_t appended_bytes, 
			    int16_t fed_buffer_dump_freq, 
			    int16_t fed_event_dump_freq, 
			    int16_t trigger_fed_id,
			    bool    using_fed_key );
  /** */
  ~SiStripRawToDigiUnpacker();

  typedef edm::DetSetVector<SiStripDigi> Digis;
  typedef edm::DetSetVector<SiStripRawDigi> RawDigis;
  
  /** Creates "pseudo" digis. */
  void createDigis( const SiStripFedCabling&,
  		    const edm::Handle<FEDRawDataCollection>&,
  		    const SiStripEventSummary&,
		    std::auto_ptr<SiStripDigiCollection>& );
  
  /** Creates "real" digis. */
  void createDigis( const SiStripFedCabling&,
		    const FEDRawDataCollection&,
		    const SiStripEventSummary&,
		    RawDigis& scope_mode,
		    RawDigis& virgin_raw,
		    RawDigis& proc_raw,
		    Digis& zero_suppr );
  
  /** */
  void triggerFed( const FEDRawDataCollection&, 
		   SiStripEventSummary& );

 private:
  
  /** Private default constructor. */
  SiStripRawToDigiUnpacker();
  
  inline void readoutOrder( uint16_t& physical_order, uint16_t& readout_order );
  inline void physicalOrder( uint16_t& readout_order, uint16_t& physical_order ); 

  inline sistrip::FedBufferFormat fedBufferFormat( const uint16_t& register_value );
  inline sistrip::FedReadoutMode fedReadoutMode( const uint16_t& register_value );
  
  void locateStartOfFedBuffer( const uint16_t& fed_id, const FEDRawData& input, FEDRawData& output );
  void dumpRawData( uint16_t fed_id, const FEDRawData&, std::stringstream& );
  
  /** Catches all possible exceptions and rethrows them as
      cms::Exception's that are caught by the framework. */ 
  void handleException( std::string method_name,
			std::string extra_info = "" ); // throw (cms::Exception);
  
 private:
  
  int16_t headerBytes_;
  int16_t fedBufferDumpFreq_;
  int16_t fedEventDumpFreq_;
  int16_t triggerFedId_;
  bool    useFedKey_;
  
  Fed9U::Fed9UEvent* fedEvent_;

  uint32_t event_;

};

// ---------- inline methods ----------

void SiStripRawToDigiUnpacker::readoutOrder( uint16_t& physical_order, 
					     uint16_t& readout_order ) {
  readout_order = ( 4*((static_cast<uint16_t>((static_cast<float>(physical_order)/8.0)))%4) +
		    static_cast<uint16_t>(static_cast<float>(physical_order)/32.0) +
		    16*(physical_order%8) );
}

void SiStripRawToDigiUnpacker::physicalOrder( uint16_t& readout_order, 
					      uint16_t& physical_order ) {
  physical_order = ( (32 * (readout_order%4)) +
		     (8 * static_cast<uint16_t>(static_cast<float>(readout_order)/4.0)) -
		     (31 * static_cast<uint16_t>(static_cast<float>(readout_order)/16.0)) );
}

sistrip::FedBufferFormat SiStripRawToDigiUnpacker::fedBufferFormat( const uint16_t& register_value ) {
  if      ( (register_value&0xF) == 0x1 ) { return sistrip::FULL_DEBUG_FORMAT; }
  else if ( (register_value&0xF) == 0x2 ) { return sistrip::APV_ERROR_FORMAT; }
  else if ( (register_value&0xF) == 0x0 ) { return sistrip::UNDEFINED_FED_BUFFER_FORMAT; }
  else                                    { return sistrip::UNKNOWN_FED_BUFFER_FORMAT; }
}

sistrip::FedReadoutMode SiStripRawToDigiUnpacker::fedReadoutMode( const uint16_t& register_value ) {
  if      ( ((register_value>>1)&0x7) == 0x0 ) { return sistrip::SCOPE_MODE; }
  else if ( ((register_value>>1)&0x7) == 0x1 ) { return sistrip::VIRGIN_RAW; }
  else if ( ((register_value>>1)&0x7) == 0x3 ) { return sistrip::PROC_RAW; }
  else if ( ((register_value>>1)&0x7) == 0x5 ) { return sistrip::ZERO_SUPPR; }
  else if ( ((register_value>>1)&0x7) == 0x6 ) { return sistrip::ZERO_SUPPR_LITE; }
  else                                         { return sistrip::UNKNOWN_FED_READOUT_MODE; }
}

#endif // EventFilter_SiStripRawToDigi_SiStripRawToDigiUnpacker_H



