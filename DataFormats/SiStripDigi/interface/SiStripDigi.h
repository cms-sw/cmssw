#ifndef DataFormats_SiStripDigi_SiStripDigi_H
#define DataFormats_SiStripDigi_SiStripDigi_H

#include "boost/cstdint.hpp"

/**  
     @brief A Digi for the silicon strip detector, containing both
     strip and adc information, and suitable for storing
     zero-suppresed hit information.
*/
class SiStripDigi {

 public:

  SiStripDigi() : strip_(0), adc_(0) {;}
  SiStripDigi( uint16_t strip, uint16_t adc ) : strip_(strip), adc_(adc) {;}
  ~SiStripDigi() {;}

  inline const uint16_t& strip()   const { return strip_; }
  inline const uint16_t& adc()     const { return adc_; }
  inline const uint16_t& channel() const { return strip(); }
 
  inline bool operator< ( const SiStripDigi& other ) const { return strip() < other.strip(); }
  
 private:
  
  uint16_t strip_;
  uint16_t adc_;

};

#endif // DataFormats_SiStripDigi_SiStripDigi_H



