#ifndef SiStripRawToDigi_H
#define SiStripRawToDigi_H

#include "CalibTracker/SiStripConnectivity/interface/SiStripConnection.h"
//
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "DataFormats/SiStripDigi/interface/StripDigi.h"
//
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
//
#include "Fed9UUtils.hh"

/**
   \class SiStripRawToDigi
   \brief Input: FEDRawDataCollection. Output: StripDigiCollection.
   \author M.Wingham, R.Bainbridge
   \version 0.1
   \date 05/09/05

   Input: FEDRawDataCollection. 
   Output: StripDigiCollection.
*/
class SiStripRawToDigi {
  
 public: // ----- public interface -----
  
  /** */
  SiStripRawToDigi( SiStripConnection& conns,
		    unsigned short verbosity = 0 );
  /** */
  ~SiStripRawToDigi();
  
  /** Takes a FEDRawDataCollection as input and creates a
      StripDigiCollection. */
  void createDigis( raw::FEDRawDataCollection& fed_buffers,
		    StripDigiCollection& digis );
  
  /** */
  inline void fedReadoutPath( std::string readout_path );
  /** */
  inline void fedReadoutMode( std::string readout_mode );
  
 private: // ----- private methods -----
  
  /** private default constructor */
  SiStripRawToDigi() {;}

  /** */
  void zeroSuppr( unsigned short fed_id,
		  StripDigiCollection& digis );
  /** */
  void scopeMode( unsigned short fed_id,
		  StripDigiCollection& digis );
  /** */
  void virginRaw( unsigned short fed_id,
		  StripDigiCollection& digis );
  /** */
  void procRaw( unsigned short fed_id, 
		StripDigiCollection& digis );
  
  /** */
  inline int readoutOrder( int physical_order );
  /** */
  inline int physicalOrder( int readout_order ); 

  /** */
  void extractHeaderInfo() {;} //@@ needs implementing!
  /** */
  void debug() {;}
  
  /** */
  void dumpFedBuffer( unsigned short fed, 
		      unsigned char* start, 
		      unsigned long size );
    
 private: // ----- private data members -----
  
  /** */
  SiStripConnection connections_;
  /** */
  unsigned short verbosity_;

  /** */
  Fed9U::Fed9UDescription* description_;
  /** */
  Fed9U::Fed9UEvent* fedEvent_;

  /** */
  std::string readoutPath_;
  /** */
  std::string readoutMode_; // unsigned short readoutMode_;

  /** FED identifiers. */
  vector<unsigned short> fedids_;

  // some debug counters
  vector<unsigned int> position_;
  vector<unsigned int> landau_;
  unsigned long nFeds_;
  unsigned long nDets_;
  unsigned long nDigis_;

};

#endif // SiStripRawToDigi_H

// inline methods

inline void SiStripRawToDigi::fedReadoutPath( std::string rpath ) {
  readoutPath_ = rpath;
}

inline void SiStripRawToDigi::fedReadoutMode( std::string rmode ) { 
  readoutMode_ = rmode;
} 

inline int SiStripRawToDigi::readoutOrder( int physical_order ) {
  return ( 4*((static_cast<int>((static_cast<float>(physical_order)/8.0)))%4) + 
	   static_cast<int>(static_cast<float>(physical_order)/32.0) + 
	   16*(physical_order%8) );
}

inline int SiStripRawToDigi::physicalOrder( int readout_order ) {
  return ( (32 * (readout_order%4)) + 
	   (8 * static_cast<int>(static_cast<float>(readout_order)/4.0)) - 
	   (31 * static_cast<int>(static_cast<float>(readout_order)/16.0)) );
}



