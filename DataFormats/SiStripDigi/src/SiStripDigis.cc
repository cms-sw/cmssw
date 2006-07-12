#include "DataFormats/SiStripDigi/interface/SiStripDigis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
//
SiStripDigis::SiStripDigis( const edm::Handle<FEDRawDataCollection>& raw_data,
			    const std::vector<uint16_t>& fed_ids,
			    const uint32_t& appended_bytes ) :
  buffers_( raw_data ),             // Reference to FED buffers
  feds_(),                          // List of FED ids 
  appendedBytes_( appended_bytes ), // Appended bytes prior to FED buffers 
  daqHdrPos_( appendedBytes_ ),     // Position of DAQ header within FEDRawData
  trkHdrPos_( daqHdrPos_ + 8 ),     // Position of tracker header
  apvErrorHdrPos_( trkHdrPos_ + 8 ),
  fullDebugHdrPos_( trkHdrPos_ + 8 ),
  apvErrorHdrSize_( 24 ),
  fullDebugHdrSize_( 16 ),
//   data_(0),
//   size_(0),
//   iterator_(0),
  error_("InvalidData")
{
  feds_.clear();
  feds_.resize( 1024, false );
  std::vector<uint16_t>::const_iterator ifed = fed_ids.begin();
  for ( ; ifed != fed_ids.end(); ifed++ ) {
    if ( *ifed < 1024 ) { feds_[*ifed] = true; }
  }
}

// -----------------------------------------------------------------------------
// 
const uint16_t SiStripDigis::invalid_ = 0xFFFF;

// -----------------------------------------------------------------------------
//
const uint16_t& SiStripDigis::adc( const uint16_t& fed_id, 
				   const uint16_t& fed_ch, 
				   const uint16_t& sample ) const { 
  static const string method = "SiStripDigis::adc";

  // Check on FED id/channel and sample
  if ( fed_id > 1023 || fed_ch > 95 || sample > 1022 ) { 
    edm::LogError(error_) << "["<<method<<"]" 
			  << "Invalid FED id/channel or sample: " 
			  << fed_id << "/" << fed_ch << "/" << sample;
    // throw
    return invalid_; 
  }
  
  if ( !feds_[fed_id] ) { 
    // throw
    return invalid_; 
  }
  
  static uint8_t* data_;
  static uint32_t size_;
  static uint32_t iterator_;

  static uint16_t last_id = 0;
  static uint16_t last_ch = 0;
  
  // Check if internal payload "pointer" is still valid
  if ( !last_id == fed_id || 
       !last_ch == fed_ch ) {
    
    // Payload "pointer" is invalid as FED id/ch have changed
    last_id = fed_id; 
    last_ch = fed_ch;

    // Retrieve 
    data_ = const_cast<uint8_t*>( buffers_->FEDData( fed_id ).data() );
    size_ = buffers_->FEDData( fed_id ).size();
  
    // Move to beginning of payload block
    if ( (data_[trkHdrPos_+1]>>4) & 0xF == 0x1 ) { // "Full debug" mode
      iterator_ = fullDebugHdrPos_ + 8*fullDebugHdrSize_; 
    } else if ( (data_[trkHdrPos_+1]>>4)&0xF == 0x2 ) { // "APV error" mode
      iterator_ = apvErrorHdrPos_ + apvErrorHdrSize_; 
    } else { 
      iterator_ = 0;
      // throw
    }
    
    // Iterate through buffer to front-end unit
    for ( uint16_t ii = 0; ii < (fed_ch%8)+1; ii++ ) {
      if ( data_[trkHdrPos_+4] & (0x1<<ii) == 0x1 ) { // Check if bit set in "FE enable" field
	iterator_ += 
	  ( (data_[iterator_+0] << 0) |  // LSB
	    (data_[iterator_+1] << 8) ); // MSB
	iterator_ = (iterator_+7) & ~7; // pads to 64-bits (8 bytes)
      }
    }
    
  } 
  
  // Check on FEDRawData pointer and size
  if ( !data_ || !size_ ) { 
    // throw
    return invalid_; 
  }
  
  // Check on payload position
  if ( iterator_ = 0 || ( iterator_ > size_ ) ) {
    // throw 
    return invalid_;
  }
  
  // Check FED readout mode using "Trk Evt_ty" field
  // (LSB, identifying "fake" mode, is ignored)
  if ( (data_[trkHdrPos_+1]>>1) & 0x7 == 0x0 ||  // Scope-mode
       (data_[trkHdrPos_+1]>>1) & 0x7 == 0x1 ||  // Virgin raw
       (data_[trkHdrPos_+1]>>1) & 0x7 == 0x3 ) { // Processed raw
    // Check sample number is less than data length
    if ( sample + 3 < ( (data_[iterator_+0] << 0) |
			(data_[iterator_+1] << 8) ) ) {
      static uint16_t adc_sample = invalid_;
      adc_sample = ( (data_[iterator_ + 3+2*sample+0] << 8) |  // MSB (byte swapped!)
		     (data_[iterator_ + 3+2*sample+1] << 0) ); // LSB (byte swapped!)
      return adc_sample;
    } else {
      // throw
      return invalid_;
    }
  }
  else if ( (data_[trkHdrPos_+1]>>1) & 0x7 == 0x5 ) { // Zero-suppr
    //iterator_ += 2;
    return invalid_;
  } 
  else if ( (data_[trkHdrPos_+1]>>1) & 0x7 == 0x6 ) { // Zero-suppr LITE
    //iterator_ += 7;
    return invalid_;
  }

  return invalid_;
  
}

