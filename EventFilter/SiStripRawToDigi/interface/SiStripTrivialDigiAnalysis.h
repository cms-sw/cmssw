#ifndef EventFilter_SiStripRawToDigi_SiStripTrivialDigiAnalysis_H
#define EventFilter_SiStripRawToDigi_SiStripTrivialDigiAnalysis_H

#include "boost/cstdint.hpp"
#include <vector>
#include <string>

using namespace std;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripTrivialDigiAnalysis.h
   @class SiStripTrivialDigiAnalysis 
   
   @brief Very simple utility class for analyzing Digis 
*/
class SiStripTrivialDigiAnalysis {
  
 public:
  
  SiStripTrivialDigiAnalysis( string class_name );
  ~SiStripTrivialDigiAnalysis();
  
  inline void addEvent()  { nEvents_++; }
  inline void addFed( uint32_t nfeds = 1 )  { nFeds_  += nfeds; }
  inline void addChan( uint32_t nchans = 1) { nChans_ += nchans; }
  inline void addDet( uint32_t ndets = 1)   { nDets_  += ndets; }

  void zsDigi( uint16_t strip, uint16_t adc );
  void vrDigi( uint16_t strip, uint16_t adc );
  void prDigi( uint16_t strip, uint16_t adc );
  void smDigi( uint16_t strip, uint16_t adc );

  void print();
  
 private:

  SiStripTrivialDigiAnalysis();

  string name_;

  vector<uint32_t> zsPos_, zsAdc_;
  vector<uint32_t> vrPos_, vrAdc_;
  vector<uint32_t> prPos_, prAdc_;
  vector<uint32_t> smPos_, smAdc_;

  uint32_t nEvents_;
  uint32_t nFeds_;
  uint32_t nChans_;
  uint32_t nDets_;

  uint32_t zsDigis_;
  uint32_t vrDigis_;
  uint32_t prDigis_;
  uint32_t smDigis_;
  
};

#endif // EventFilter_SiStripRawToDigi_SiStripTrivialDigiAnalysis_H

