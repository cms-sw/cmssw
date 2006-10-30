#include "DataFormats/SiStripDigi/interface/SiStripDigiCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
// 
const uint16_t SiStripDigiCollection::invalid_ = 0xFFFF;

// -----------------------------------------------------------------------------
//
SiStripDigiCollection::SiStripDigiCollection( const edm::Handle<FEDRawDataCollection>& buffers,
					      const vector<uint16_t>& fed_ids, 
					      const vector<sistrip::FedBufferFormat>& format,
					      const vector<sistrip::FedReadoutMode>& mode,
					      const vector<uint8_t>& fe_enable_bits,
					      const vector<uint16_t>& appended_bytes ) :
  buffers_(buffers),
  feds_( vector<bool>( 1+FEDNumbering::lastFEDId(), false ) ),
  data_( vector<uint8_t*>( 1+FEDNumbering::lastFEDId(), static_cast<uint8_t*>(0) ) ),
  size_( vector<uint32_t>( 1+FEDNumbering::lastFEDId(), 0 ) ),
  readoutMode_( vector<sistrip::FedReadoutMode>( 1+FEDNumbering::lastFEDId(), sistrip::UNDEFINED_FED_READOUT_MODE ) ),
  readoutPath_( vector<sistrip::FedReadoutPath>( 1+FEDNumbering::lastFEDId(), sistrip::UNDEFINED_FED_READOUT_PATH ) ),
  payload_( 1+FEDNumbering::lastFEDId(), vector<uint16_t>( sistrip::FEUNITS_PER_FED, 0 ) ),
  error_("InvalidData") 
{
 
  // Check vector size
  if ( format.size() != static_cast<uint16_t>(1+FEDNumbering::lastFEDId()) ) {
    stringstream ss;
    ss << "["<<__PRETTY_FUNCTION__<<"]" 
       << "Unexpected size for vector of FedBufferFormat's! (" << format.size() 
       << "). Resizing to " << 1+FEDNumbering::lastFEDId() << "...";
    edm::LogWarning(error_) << ss.str();
    const_cast<vector<sistrip::FedBufferFormat>&>(format).resize( 1+FEDNumbering::lastFEDId(), sistrip::APV_ERROR_FORMAT );
  }
  
  // Check vector size
  if ( mode.size() != static_cast<uint16_t>(1+FEDNumbering::lastFEDId()) ) {
    stringstream ss;
    ss << "["<<__PRETTY_FUNCTION__<<"]" 
       << "Unexpected size for vector of FedReadoutMode's! (" << mode.size() 
       << "). Resizing to " << 1+FEDNumbering::lastFEDId() << "...";
    edm::LogWarning(error_) << ss.str();
    const_cast<vector<sistrip::FedReadoutMode>&>(mode).resize( 1+FEDNumbering::lastFEDId(), sistrip::ZERO_SUPPR );
  }
  
  // Iterate through FED ids
  std::vector<uint16_t>::const_iterator ifed = fed_ids.begin();
  for ( ; ifed != fed_ids.end(); ifed++ ) {

    // Check FED id is valid
    if ( *ifed < 1+FEDNumbering::lastFEDId() ) { 

      // Flag FED id is to be used
      feds_[*ifed] = true; 
      
      // Retrieve pointer to FED data
      data_[*ifed] = const_cast<uint8_t*>( buffers_->FEDData( *ifed ).data() );
      size_[*ifed] = buffers_->FEDData( *ifed ).size();
      
      // Retrieve readout path of FED buffers (using FED trailer info)
      uint32_t* buffer_len = reinterpret_cast<uint32_t*>( &data_[*ifed][size_[*ifed]-8] );
      uint32_t buffer_size = ( size_[*ifed]-appended_bytes[*ifed] + 7 ) & ~7;
      if ( buffer_len[0] == ( 0xA0000000 | (buffer_size>>3) ) ) {
	readoutPath_[*ifed] = sistrip::VME_READOUT;
      } else if ( buffer_len[1] == ( 0xA0000000 | (buffer_size>>3) ) ) {
	readoutPath_[*ifed] = sistrip::SLINK_READOUT;
      } else { 
	readoutPath_[*ifed] = sistrip::UNKNOWN_FED_READOUT_PATH; 
      }
      //       cout << " buffer_len[0]: 0x" 
      // 	   << hex << setw(8) << setfill('0') << static_cast<uint32_t>( buffer_len[0] ) << " dec"
      // 	   << dec << static_cast<uint32_t>( buffer_len[0] ) && 0xFFFFFF
      // 	   << " buffer_len[1]: 0x" 
      // 	   << hex << setw(8) << setfill('0') << static_cast<uint32_t>( buffer_len[1] ) << " dec"
      // 	   << dec << static_cast<uint32_t>( buffer_len[1] ) && 0xFFFFFF
      // 	   << " buffer_size: " 
      // 	   << hex << setw(8) << setfill('0') << (buffer_size>>3) << dec 
      // 	   << " readout path: " << readoutPath_[*ifed]
      // 	   << endl;
      
      // Set position of payload blocks within FE units of FED buffer
      if ( format[*ifed] == sistrip::APV_ERROR_FORMAT )  { 
	payload_[*ifed].clear();
	payload_[*ifed].resize( sistrip::FEUNITS_PER_FED, appended_bytes[*ifed] + sistrip::DAQ_HDR_SIZE + sistrip::TRK_HDR_SIZE + sistrip::APV_ERROR_HDR_SIZE );
      } else if ( format[*ifed] == sistrip::FULL_DEBUG_FORMAT ) { 
	payload_[*ifed].clear();
	payload_[*ifed].resize( sistrip::FEUNITS_PER_FED, appended_bytes[*ifed] + sistrip::DAQ_HDR_SIZE + sistrip::TRK_HDR_SIZE + sistrip::FULL_DEBUG_HDR_SIZE );
	uint16_t fe_offset = 0;
	for ( uint16_t ii = 0; ii < sistrip::FEUNITS_PER_FED; ii++ ) {
	  uint16_t index = payload_[*ifed][ii] - sistrip::FULL_DEBUG_HDR_SIZE + sistrip::FE_HDR_SIZE*ii;
	  uint16_t index0 = 0;
	  uint16_t index1 = 0;
	  if ( readoutPath_[*ifed] == sistrip::VME_READOUT ) { 
	    index0 = index + swap32(14); 
	    index1 = index + swap32(15); 
	  } else { 
	    index0 = index + 14; 
	    index0 = index + 15; 
	  }
	  
	  uint16_t fe_length = static_cast<uint16_t>( ((data_[*ifed][index0]&0xFF)<<0) | ((data_[*ifed][index1]&0xFF)<<8) );
	  //uint16_t fe_padded = static_cast<uint16_t>( (fe_length+7)&~7 );
	  // 	  cout << "["<<__PRETTY_FUNCTION__<<"]"
	  // 	       << " FED id: " << *ifed
	  // 	       << " FE unit: " << ii
	  // 	       << " FE enable bits: 0x" << hex << setw(8) << setfill('0') << static_cast<uint16_t>( fe_enable_bits[*ifed] ) << dec
	  // 	       << " FE enabled?: " << static_cast<uint16_t>( fe_enable_bits[*ifed] & (0x1<<ii) )
	  // 	       << " FE start: " << payload_[*ifed][ii]
	  // 	       << " index/index0/index1: " <<  index << "/" << index0 << "/" << index1
	  // 	       << " FE length: " << fe_length
	  // 	       << " FE padded: " << fe_padded;
	  if ( fe_enable_bits[*ifed] & (0x1<<ii) ) {
	    payload_[*ifed][ii] += fe_offset;
	    fe_offset += fe_length; 
	    // 	    cout << " FE unit " << ii << " is used! ";
	  } else { payload_[*ifed][ii] = 0; }
	  // 	  cout << " FE start (new): " << payload_[*ifed][ii] 
	  // 	       << endl;
	}
      } else if ( format[*ifed] == sistrip::UNDEFINED_FED_BUFFER_FORMAT ) { 
	//@@ anything here?
      } else {
	stringstream ss;
	ss << "["<<__PRETTY_FUNCTION__<<"]"
	   << " Unexpected value for FedBufferformat! (" << format[*ifed]
	   << "). Ignoring FED with id " << *ifed << "...";
	edm::LogWarning(error_) << ss.str();
	feds_[*ifed] = false; 
      }
      
      // Set FED readout modes
      readoutMode_[*ifed] = mode[*ifed];
      
    } else {
      stringstream ss;
      ss << "["<<__PRETTY_FUNCTION__<<"]"
	 << " Unexpected FED id! (" << *ifed
	 << "). Allowed range is 0->"
	 << ( FEDNumbering::lastFEDId() );
      edm::LogWarning(error_) << ss.str();
    }
  }

}

// -----------------------------------------------------------------------------
//
const uint16_t& SiStripDigiCollection::adc( const uint16_t& fed_id, 
					    const uint16_t& fed_ch, 
					    const uint16_t& sample ) const { 

  // Check on FED id/channel and sample
  if ( fed_id > 1023 || fed_ch > 95 || sample > 1022 ) { 
    edm::LogWarning(error_) << "["<<__PRETTY_FUNCTION__<<"]" 
			    << "Invalid FED id/channel or sample: " 
			    << fed_id << "/" << fed_ch << "/" << sample;
    return invalid_; 
  }
  
  if ( !feds_[fed_id] ) { 
    edm::LogWarning(error_) << "["<<__PRETTY_FUNCTION__<<"]" 
			    << "FED id " << fed_id << "is not available!";
    return invalid_; 
  }
  
  static uint16_t last_id = 0;
  static uint16_t last_ch = 0;
  
  // Check if internal payload "pointer" is still valid
  if ( !last_id == fed_id || 
       !last_ch == fed_ch ) {
    
    // Payload "pointer" is invalid as FED id/ch have changed
    last_id = fed_id; 
    last_ch = fed_ch;
        
    // Check on FEDRawData pointer and size
    if ( !data_[fed_id] || !size_[fed_id] ) { 
      // throw
      return invalid_; 
    }

    // Check on payload position
    if ( payload_[fed_id][fed_ch%8] == 0 || 
	 payload_[fed_id][fed_ch%8] > size_[fed_id] ) {
      // throw 
      return invalid_;
    }
    
  }
  
  if ( readoutMode_[fed_id] == sistrip::SCOPE_MODE ||
       readoutMode_[fed_id] == sistrip::VIRGIN_RAW ||
       readoutMode_[fed_id] == sistrip::PROC_RAW ) {
    
    static uint16_t temporary;
    
    if ( readoutPath_[fed_id] == sistrip::VME_READOUT ) { 
      temporary = ( (data_[fed_id][ swap32(payload_[fed_id][fed_ch%8]+0) ] << 0) |
		    (data_[fed_id][ swap32(payload_[fed_id][fed_ch%8]+1) ] << 8) );
    } else {
      temporary = ( (data_[fed_id][ payload_[fed_id][fed_ch%8]+0 ] << 0) |
		    (data_[fed_id][ payload_[fed_id][fed_ch%8]+1 ] << 8) );
    }      
    
    // Check sample number is less than data length
    if ( sample + 3 < temporary ) {
      static uint16_t adc_sample = invalid_;
      if ( readoutPath_[fed_id] == sistrip::VME_READOUT ) { 
	adc_sample = ( (data_[fed_id][ swap32( payload_[fed_id][fed_ch%8]+3+2*sample+0) ] << 8) |  // MSB (byte swapped!)
		       (data_[fed_id][ swap32( payload_[fed_id][fed_ch%8]+3+2*sample+1) ] << 0) ); // LSB (byte swapped!)
	if ( sample < 3 ) {
	  // 	  cout << " FedId: " << fed_id
	  // 	       << " FedCh: " << fed_ch
	  // 	       << " sample: " << sample
	  // 	       << " offset0: " << (payload_[fed_id][fed_ch%8]+3+2*sample+0)
	  // 	       << " offset1: " << (payload_[fed_id][fed_ch%8]+3+2*sample+1)
	  // 	       << " swapped0: " << swap32(payload_[fed_id][fed_ch%8]+3+2*sample+0)
	  // 	       << " swapped1: " << swap32(payload_[fed_id][fed_ch%8]+3+2*sample+1)
	  // 	       << " data0: " << static_cast<uint16_t>( data_[fed_id][ swap32( payload_[fed_id][fed_ch%8]+3+2*sample+0) ] )
	  // 	       << " data1: " << static_cast<uint16_t>( data_[fed_id][ swap32( payload_[fed_id][fed_ch%8]+3+2*sample+1) ] )
	  // 	       << " data: " << adc_sample
	  // 	       << endl;
	}
      } else {
	adc_sample = ( (data_[fed_id][ payload_[fed_id][fed_ch%8]+3+2*sample+0 ] << 8) |  // MSB (byte swapped!)
		       (data_[fed_id][ payload_[fed_id][fed_ch%8]+3+2*sample+1 ] << 0) ); // LSB (byte swapped!)
      }
      //       if ( !sample ) {
      // 	cout << "["<<__PRETTY_FUNCTION__<<"]"
      // 	     << " FED id: " << fed_id
      // 	     << " FED ch: " << fed_ch
      // 	     << " sample: " << sample
      // 	     << " iterator: " << (payload_[fed_id][fed_ch%8]+3+2*sample)
      // 	     << endl;
      //       }
      return adc_sample;
    } else {
      // throw
      return invalid_;
    }


  } else if ( readoutMode_[fed_id] == sistrip::ZERO_SUPPR ) {

    //iterator_ += 2;
    return invalid_;

  } else if ( readoutMode_[fed_id] == sistrip::ZERO_SUPPR_LITE ) {

    //iterator_ += 7;
    return invalid_;
    
  }
  
  return invalid_;
  
}

// -----------------------------------------------------------------------------
//

