// Last commit: $Id: SiStripDigiToRaw.h,v 1.19 2009/03/27 10:21:52 bainbrid Exp $

#ifndef EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H
#define EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "boost/cstdint.hpp"
#include <string>

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferGenerator.h"

class SiStripFedCabling;
class FEDRawDataCollection;
class SiStripDigi;

namespace sistrip {

  /**
     @file EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h
     @class sistrip::DigiToRaw 
   
     @brief Input: edm::DetSetVector<SiStripDigi>. 
     Output: FEDRawDataCollection.
  */
  class DigiToRaw {
    
  public: // ----- public interface -----
    
    DigiToRaw( std::string readout_mode, 
	       int16_t appended_bytes,
	       bool use_fed_key );
    ~DigiToRaw();
    
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
    FEDBufferGenerator bufferGenerator_;
    
  };
  
}


////////////////////////////////////////////////////////////////////////////////
//@@@ TO BE DEPRECATED BELOW!!!


/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h
   @class OldSiStripDigiToRaw 
   
   @brief Input: edm::DetSetVector<SiStripDigi>. 
   Output: FEDRawDataCollection.
*/
class OldSiStripDigiToRaw {
  
 public: // ----- public interface -----
  
  OldSiStripDigiToRaw( std::string readout_mode, 
		    int16_t appended_bytes,
		    bool use_fed_key );
  ~OldSiStripDigiToRaw();
  
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

