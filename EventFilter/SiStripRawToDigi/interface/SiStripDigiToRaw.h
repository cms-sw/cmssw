#ifndef EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H
#define EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "Fed9UUtils.hh"
#include "boost/cstdint.hpp"
#include <string>
#include <vector>

class SiStripFedCabling;
class FEDRawDataCollection;
class SiStripDigi;

using namespace std;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h
   @class SiStripDigiToRaw 
   
   @brief Input: edm::DetSetVector<SiStripDigi>. 
   Output: FEDRawDataCollection.
*/
class SiStripDigiToRaw {
  
 public: // ----- public interface -----
  
  SiStripDigiToRaw( string readout_mode, 
		    int16_t appended_bytes );
  ~SiStripDigiToRaw();
  
  void createFedBuffers( edm::ESHandle<SiStripFedCabling>& cabling,
			 edm::Handle< edm::DetSetVector<SiStripDigi> >& digis,
			 auto_ptr<FEDRawDataCollection>& buffers );
  
  inline void fedReadoutMode( string mode )     { readoutMode_ = mode; }
  inline void nAppendedBytes( uint16_t nbytes ) { nAppendedBytes_ = nbytes; }
  
 private: // ----- private data members -----

  string readoutMode_;
  uint16_t nAppendedBytes_;

  // some debug counters
  vector<unsigned int> position_;
  vector<unsigned int> landau_;
  unsigned long nFeds_;
  unsigned long nDigis_;

};

#endif // EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H

