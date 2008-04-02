// Last commit: $Id: SiStripDigiToRaw.h,v 1.16 2007/10/09 17:03:27 bainbrid Exp $

#ifndef EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H
#define EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
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
		    int16_t appended_bytes,
		    bool use_fed_key );
  ~SiStripDigiToRaw();
  
  void createFedBuffers( edm::Event&, 
			 edm::ESHandle<SiStripFedCabling>& cabling,
			 edm::Handle< edm::DetSetVector<SiStripDigi> >& digis,
			 std::auto_ptr<FEDRawDataCollection>& buffers );
  
  inline void fedReadoutMode( std::string mode ) { readoutMode_ = mode; }
  inline void nAppendedBytes( uint16_t nbytes ) { nAppendedBytes_ = nbytes; }
  
 private: // ----- private data members -----

  std::string readoutMode_;
  uint16_t nAppendedBytes_;
  bool useFedKey_;

};

#endif // EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H

