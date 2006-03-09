#ifndef DataFormats_SiStripDigi_SiStripRawDigi_H
#define DataFormats_SiStripDigi_SiStripRawDigi_H

#include "boost/cstdint.hpp"

/** 
    @brief A Digi for the silicon strip detector, containing only adc
    information, and suitable for storing raw hit information.
*/
class SiStripRawDigi {

 public:

  SiStripRawDigi( uint16_t adc ) : adc_(adc) {;}
  SiStripRawDigi() : adc_(0) {;}
  ~SiStripRawDigi() {;}

  inline const uint16_t& adc()     const { return adc_; }
  
  inline bool operator< ( const SiStripRawDigi& other ) const { return true; }
  
 private:
  
  uint16_t adc_;
  
};

#endif // DataFormats_SiStripDigi_SiStripRawDigi_H



