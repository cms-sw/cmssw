#ifndef EventFilter_SiStripRawToDigi_SiStripTrivialDigiAnalysis_H
#define EventFilter_SiStripRawToDigi_SiStripTrivialDigiAnalysis_H

#include "boost/cstdint.hpp"
#include <sstream>
#include <vector>

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripTrivialDigiAnalysis.h
   @class SiStripTrivialDigiAnalysis 
   @brief Very simple utility class for analyzing Digis 
*/
class SiStripTrivialDigiAnalysis {
  
 public:

  

  /** Default constructor. */  
  SiStripTrivialDigiAnalysis() : 
    events_(0), 
    feds_(0), 
    channels_(0), 
    strips_(0),
    digis_(0), 
    size_(1024), 
    pos_(size_+1,0), 
    adc_(size_+1,0) {;}
  
  /** Pipes collected statistics to stringstream. */
  void print( std::stringstream& );

  // setters
  inline void pos( const uint16_t& pos );
  inline void adc( const uint16_t& adc );
  
  // getters
  inline const std::vector<uint16_t>& pos();
  inline const std::vector<uint16_t>& adc();
  
 public:

  uint32_t events_;
  uint32_t feds_;
  uint32_t channels_;
  uint32_t strips_;
  uint32_t digis_;
  
  const uint16_t size_;
  
 private:
  
  std::vector<uint16_t> pos_;
  std::vector<uint16_t> adc_;

};

// ---------- inline methods ----------

void SiStripTrivialDigiAnalysis::pos( const uint16_t& pos ) {
  if ( pos < size_ ) { pos_[pos]++; }
  else { pos_[size_]++; } 
}

void SiStripTrivialDigiAnalysis::adc( const uint16_t& adc ) {
  if ( adc < size_ ) { adc_[adc]++; }
  else { adc_[size_]++; } 
}

const std::vector<uint16_t>& SiStripTrivialDigiAnalysis::pos() { return pos_; }

const std::vector<uint16_t>& SiStripTrivialDigiAnalysis::adc() { return adc_; }

#endif // EventFilter_SiStripRawToDigi_SiStripTrivialDigiAnalysis_H

