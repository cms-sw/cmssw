#ifndef DataFormats_SiStripDigi_SiStripDigis_H
#define DataFormats_SiStripDigi_SiStripDigis_H

#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>

/**  
     @brief 
*/
class SiStripDigis {
  
 public:

  SiStripDigis( const edm::Handle<FEDRawDataCollection>& raw_data,
		const std::vector<uint16_t>& fed_ids, 
		const uint32_t& appended_bytes = 0 );
  
  SiStripDigis() {;}
  ~SiStripDigis() {;}
  
  /** All methods return "invalid" (0xFFFF) if there are problems
      retrieving the ADC value for a given FED id/chan and sample. */
  static const uint16_t invalid_;
  
  static const uint16_t stripPerFedChannel_;
  
  const uint16_t& adc( const uint16_t& fed_id, 
		       const uint16_t& fed_ch, 
		       const uint16_t& sample ) const;
  
 private:

  /** Position of DAQ trailer. */
  inline const uint16_t& daqTrailer() { static uint16_t tmp = 0; return tmp; } 

  /** */
  edm::RefProd<FEDRawDataCollection> buffers_;

  /** */
  std::vector<bool> feds_;
  
  /** Number of appended bytes prior to each FED buffer. */
  const uint32_t appendedBytes_;
  
  /** Position of DAQ header. */
  const uint32_t daqHdrPos_;
  
  /** Position of tracker-specific "special" header. */
  const uint32_t trkHdrPos_;
  
  /** Position of payload when operating in "APV error" mode. */
  const uint32_t apvErrorHdrPos_;
  
  /** Position of payload when operating in "full debug" mode. */
  const uint32_t fullDebugHdrPos_; 

  const uint16_t apvErrorHdrSize_;
  const uint16_t fullDebugHdrSize_; 

/*   unsigned char* data_; */
/*   uint32_t size_; */
/*   uint32_t iterator_; */

  const std::string error_;

};

#endif // DataFormats_SiStripDigi_SiStripDigis_H




