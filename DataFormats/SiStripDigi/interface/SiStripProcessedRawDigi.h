#ifndef AnalysisDataFormats_SiStripDigi_SiStripProcessedRawDigi_H
#define AnalysisDataFormats_SiStripDigi_SiStripProcessedRawDigi_H

#include "DataFormats/Common/interface/traits.h"

/** 
    @brief A signed Digi for the silicon strip detector, containing
    only adc information, and suitable for storing processed
    (pedestal, cmn subtracted) hit information. NOTA BENE: these digis
    use the DetSetVector, but the public inheritence from
    edm::DoNotSortUponInsertion ensures that the digis are NOT sorted
    by the DetSetVector::post_insert() method. The strip position is
    therefore inferred from the position of the digi within its
    container (the DetSet private vector).
*/
class SiStripProcessedRawDigi : public edm::DoNotSortUponInsertion {
  
 public:

  SiStripProcessedRawDigi( const float& adc ) : adc_(adc) {;}

  SiStripProcessedRawDigi() : adc_(0) {;}
  ~SiStripProcessedRawDigi() {;}
  
  inline const float& adc() const;
  
  /** Not used! (even if implementation is required). */
  inline bool operator< ( const SiStripProcessedRawDigi& other ) const;
  
 private:
  
  float adc_;
  
};

#include<iostream>
inline std::ostream & operator<<(std::ostream & o, const SiStripProcessedRawDigi& digi) {
  return o << " " << digi.adc();
}

// inline methods 
const float& SiStripProcessedRawDigi::adc() const {  return adc_; }
bool SiStripProcessedRawDigi::operator< ( const SiStripProcessedRawDigi& other ) const { return ( this->adc() < other.adc() ); }

#endif // AnalysisDataFormats_SiStripDigi_SiStripProcessedRawDigi_H

