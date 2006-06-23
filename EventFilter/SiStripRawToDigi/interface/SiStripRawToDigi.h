#ifndef EventFilter_SiStripRawToDigi_SiStripRawToDigi_H
#define EventFilter_SiStripRawToDigi_SiStripRawToDigi_H

#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Handle.h>
#include "DataFormats/Common/interface/DetSetVector.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripTrivialDigiAnalysis.h"
#include "Fed9UUtils.hh"
#include "boost/cstdint.hpp"
#include <ostream>
 
class FEDRawDataCollection;
class FEDRawData;
class SiStripDigi;
class SiStripRawDigi;
class SiStripEventSummary;
class SiStripFedCabling;

using namespace std;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripRawToDigi.h
   @class SiStripRawToDigi 
   
   @brief Input: FEDRawDataCollection. 
   Output: edm::DetSetVector<SiStripDigi>.
   
*/
class SiStripRawToDigi {
  
 public: // ----- public interface -----
  
  SiStripRawToDigi( int16_t appended_bytes, 
		    int16_t dump_frequency, 
		    bool use_det_id,
		    uint16_t trigger_fed_id );
  ~SiStripRawToDigi();
  
  
  void createDigis( const uint32_t& event,
		    edm::ESHandle<SiStripFedCabling>& cabling, //@@ pass-by-ref otherwise auto_ptr "sink"?! 
		    edm::Handle<FEDRawDataCollection>& buffers,
		    auto_ptr< edm::DetSetVector<SiStripRawDigi> >& scope_mode,
		    auto_ptr< edm::DetSetVector<SiStripRawDigi> >& virgin_raw,
		    auto_ptr< edm::DetSetVector<SiStripRawDigi> >& proc_raw,
		    auto_ptr< edm::DetSetVector<SiStripDigi> >& zero_suppr,
		    auto_ptr< SiStripEventSummary >& summary );
  
 private: // ----- private methods -----
  
  inline void readoutOrder( uint16_t& physical_order, uint16_t& readout_order );
  inline void physicalOrder( uint16_t& readout_order, uint16_t& physical_order ); 
  
  void triggerFed( const FEDRawData& trigger_fed,
		   auto_ptr< SiStripEventSummary >& summary );
  void locateStartOfFedBuffer( uint16_t fed_id, const FEDRawData& input, FEDRawData& output );
  void dumpRawData( uint16_t fed_id, const FEDRawData&, std::ostream& );
  void digiInfo( vector<uint32_t>& det_ids, //@@ TEMPORARY!
		 auto_ptr< edm::DetSetVector<SiStripRawDigi> >& scope_mode,
		 auto_ptr< edm::DetSetVector<SiStripRawDigi> >& virgin_raw,
		 auto_ptr< edm::DetSetVector<SiStripRawDigi> >& proc_raw,
		 auto_ptr< edm::DetSetVector<SiStripDigi> >& zero_suppr );
  
  /** Catches all possible exceptions and rethrows them as
      cms::Exception's that are caught by the framework. */ 
  void handleException( const string& method_name,
			const string& extra_info = "" ) throw (cms::Exception);
  
 private: // ----- private data members -----
  
  Fed9U::Fed9UEvent* fedEvent_;
  Fed9U::Fed9UDescription* fedDescription_;

  int16_t headerBytes_;
  int16_t dumpFrequency_;
  bool useFedKey_;
  uint16_t triggerFedId_;

  SiStripTrivialDigiAnalysis anal_;

  vector<uint16_t> skews_; //@@ debug
  
};

void SiStripRawToDigi::readoutOrder( uint16_t& physical_order, 
				     uint16_t& readout_order ) {
  readout_order = ( 4*((static_cast<uint16_t>((static_cast<float>(physical_order)/8.0)))%4) +
		    static_cast<uint16_t>(static_cast<float>(physical_order)/32.0) +
		    16*(physical_order%8) );
}

void SiStripRawToDigi::physicalOrder( uint16_t& readout_order, 
				      uint16_t& physical_order ) {
  physical_order = ( (32 * (readout_order%4)) +
		     (8 * static_cast<uint16_t>(static_cast<float>(readout_order)/4.0)) -
		     (31 * static_cast<uint16_t>(static_cast<float>(readout_order)/16.0)) );
}

#endif // EventFilter_SiStripRawToDigi_SiStripRawToDigi_H



