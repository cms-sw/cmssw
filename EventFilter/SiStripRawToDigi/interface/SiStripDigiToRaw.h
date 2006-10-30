#ifndef EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H
#define EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripTrivialDigiAnalysis.h"
#include "Fed9UUtils.hh"
#include "boost/cstdint.hpp"
#include <string>

class SiStripFedCabling;
class FEDRawDataCollection;
class SiStripDigi;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h
   @class SiStripDigiToRaw 
   
   @brief Input: edm::DetSetVector<SiStripDigi>. 
   Output: FEDRawDataCollection.
*/
class SiStripDigiToRaw {
  
 public: // ----- public interface -----
  
  SiStripDigiToRaw( std::string readout_mode, 
		    int16_t appended_bytes );
  ~SiStripDigiToRaw();
  
  void createFedBuffers( edm::ESHandle<SiStripFedCabling>& cabling,
			 edm::Handle< edm::DetSetVector<SiStripDigi> >& digis,
			 std::auto_ptr<FEDRawDataCollection>& buffers );
  
  inline void fedReadoutMode( std::string mode ) { readoutMode_ = mode; }
  inline void nAppendedBytes( uint16_t nbytes ) { nAppendedBytes_ = nbytes; }
  
 private: // ----- private data members -----

  std::string readoutMode_;
  uint16_t nAppendedBytes_;

  SiStripTrivialDigiAnalysis anal_;

};

#endif // EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H

