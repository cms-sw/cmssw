// Last commit: $Id: SiStripDigiToRaw.h,v 1.23 2009/09/14 14:01:03 nc302 Exp $

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
class SiStripRawDigi;

namespace sistrip {

  /**
     @file EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h
     @class sistrip::DigiToRaw 
   
     @brief Input: edm::DetSetVector<SiStripDigi>. 
     Output: FEDRawDataCollection.
  */
  class DigiToRaw {
    
  public: // ----- public interface -----
    
    DigiToRaw( FEDReadoutMode, bool use_fed_key );
    ~DigiToRaw();
    
    void createFedBuffers( edm::Event&, 
			   edm::ESHandle<SiStripFedCabling>& cabling,
			   edm::Handle< edm::DetSetVector<SiStripDigi> >& digis,
			   std::auto_ptr<FEDRawDataCollection>& buffers );
    void createFedBuffers( edm::Event&, 
			   edm::ESHandle<SiStripFedCabling>& cabling,
			   edm::Handle< edm::DetSetVector<SiStripRawDigi> >& digis,
			   std::auto_ptr<FEDRawDataCollection>& buffers);
    
    inline void fedReadoutMode( FEDReadoutMode mode ) { mode_ = mode; }

  private: // ----- private data members -----
    
    template<class Digi_t>
    void createFedBuffers_( edm::Event&, 
			    edm::ESHandle<SiStripFedCabling>& cabling,
			    edm::Handle< edm::DetSetVector<Digi_t> >& digis,
			    std::auto_ptr<FEDRawDataCollection>& buffers,
			    bool zeroSuppressed);
    uint16_t STRIP(const edm::DetSet<SiStripDigi>::const_iterator& it, const edm::DetSet<SiStripDigi>::const_iterator& begin) const;
    uint16_t STRIP(const edm::DetSet<SiStripRawDigi>::const_iterator& it, const edm::DetSet<SiStripRawDigi>::const_iterator& begin) const;
    
    FEDReadoutMode mode_;
    bool useFedKey_;
    FEDBufferGenerator bufferGenerator_;
    
  };
  
}


#endif // EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H

