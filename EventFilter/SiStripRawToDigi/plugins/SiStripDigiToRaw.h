
#ifndef EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H
#define EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "boost/cstdint.hpp"
#include <string>

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferGenerator.h"
#include "WarningSummary.h"

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
  class dso_hidden DigiToRaw {
    
  public: // ----- public interface -----
    
    DigiToRaw(FEDReadoutMode mode, uint8_t packetCode, bool use_fed_key);
    ~DigiToRaw();
    
    //digi to raw with default headers
    void createFedBuffers( edm::Event&, 
			   edm::ESHandle<SiStripFedCabling>& cabling,
			   edm::Handle< edm::DetSetVector<SiStripDigi> >& digis,
			   std::unique_ptr<FEDRawDataCollection>& buffers );
    void createFedBuffers( edm::Event&, 
			   edm::ESHandle<SiStripFedCabling>& cabling,
			   edm::Handle< edm::DetSetVector<SiStripRawDigi> >& digis,
			   std::unique_ptr<FEDRawDataCollection>& buffers);

    //with input raw data for copying header   
    void createFedBuffers( edm::Event&, 
			   edm::ESHandle<SiStripFedCabling>& cabling,
			   edm::Handle<FEDRawDataCollection> & rawbuffers,
			   edm::Handle< edm::DetSetVector<SiStripDigi> >& digis,
			   std::unique_ptr<FEDRawDataCollection>& buffers );
    void createFedBuffers( edm::Event&, 
			   edm::ESHandle<SiStripFedCabling>& cabling,
			   edm::Handle<FEDRawDataCollection> & rawbuffers,
			   edm::Handle< edm::DetSetVector<SiStripRawDigi> >& digis,
			   std::unique_ptr<FEDRawDataCollection>& buffers);
    
    inline void fedReadoutMode( FEDReadoutMode mode ) { mode_ = mode; }

    void printWarningSummary() const { warnings_.printSummary(); }

  private: // ----- private data members -----
    
    template<class Digi_t>
    void createFedBuffers_( edm::Event&, 
			    edm::ESHandle<SiStripFedCabling>& cabling,
			    edm::Handle< edm::DetSetVector<Digi_t> >& digis,
			    std::unique_ptr<FEDRawDataCollection>& buffers,
			    bool zeroSuppressed);

    template<class Digi_t>
    void createFedBuffers_( edm::Event&, 
			    edm::ESHandle<SiStripFedCabling>& cabling,
			    edm::Handle<FEDRawDataCollection> & rawbuffers,
			    edm::Handle< edm::DetSetVector<Digi_t> >& digis,
			    std::unique_ptr<FEDRawDataCollection>& buffers,
			    bool zeroSuppressed);



    uint16_t STRIP(const edm::DetSet<SiStripDigi>::const_iterator& it, const edm::DetSet<SiStripDigi>::const_iterator& begin) const;
    uint16_t STRIP(const edm::DetSet<SiStripRawDigi>::const_iterator& it, const edm::DetSet<SiStripRawDigi>::const_iterator& begin) const;
    
    FEDReadoutMode mode_;
    uint8_t packetCode_;
    bool useFedKey_;
    FEDBufferGenerator bufferGenerator_;

    WarningSummary warnings_;
  };
  
}


#endif // EventFilter_SiStripRawToDigi_SiStripDigiToRaw_H

