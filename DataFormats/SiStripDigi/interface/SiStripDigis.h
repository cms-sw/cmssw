#ifndef DataFormats_SiStripDigi_SiStripDigis_H
#define DataFormats_SiStripDigi_SiStripDigis_H

#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>

/**  
     @brief 
*/
class SiStripDigis {
  
 public:
  
  SiStripDigis( const edm::Handle<FEDRawDataCollection>&,
		const std::vector<uint16_t>& fed_ids, 
		const std::vector<sistrip::FedBufferFormat>&,
		const std::vector<sistrip::FedReadoutMode>&,
		const std::vector<uint8_t>& fe_enable_bits,
		const std::vector<uint16_t>& appended_bytes );
  
  SiStripDigis() {;}
  ~SiStripDigis() {;}
  
  /** All methods return "invalid" (0xFFFF) if there are problems
      retrieving the ADC value for a given FED id/chan and sample. */
  static const uint16_t invalid_;
  
  const uint16_t& adc( const uint16_t& fed_id, 
		       const uint16_t& fed_ch, 
		       const uint16_t& sample ) const;
  
 private:

  /** Returns iterator. */
  inline uint16_t swap32( const uint16_t& byte ) const; 
  
  /** Reference to collection of FEDRawData objects. */
  edm::RefProd<FEDRawDataCollection> buffers_;
  
  /** FED ids of FEDRawData objects that have data. */
  std::vector<bool> feds_;
    
  /** Pointer to data of FEDRawData object. */
  std::vector<uint8_t*> data_;
  
  /** Number of bytes within FEDRawData object. */
  std::vector<uint32_t> size_;
  
  /** Readout mode for all FED ids. */
  std::vector<sistrip::FedReadoutMode> readoutMode_;
  
  /** Readout path for all FED ids. */
  std::vector<sistrip::FedReadoutPath> readoutPath_; 
  
  /** Payload position for individual FE units within a FED buffer. */
  std::vector< std::vector<uint16_t> > payload_;
  
  /** String defining error category. */
  std::string error_;
  
};

// ---------- inline methods ----------

uint16_t SiStripDigis::swap32( const uint16_t& byte ) const { 
  return ( ((byte/8)*8) + (((7-(byte%8))+4)%8) ); // ((byte+4)%8) ); 
}

#endif // DataFormats_SiStripDigi_SiStripDigis_H




