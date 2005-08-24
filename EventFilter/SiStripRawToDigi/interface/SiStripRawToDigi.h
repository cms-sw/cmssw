#ifndef SiStripRawToDigi_H
#define SiStripRawToDigi_H

#include "CalibTracker/SiStripConnectivity/interface/SiStripConnection.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
//
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
//
#include "Fed9UUtils.hh"

/**
   \class SiStripRawToDigi
   \brief Takes a FEDRawDataCollection as input and creates a
   StripDigiCollection.
   \author M.Wingham, R.Bainbridge
   \version 0.1
   \date 09/08/05

   Takes a FEDRawDataCollection as input and creates a
   StripDigiCollection.
*/
class SiStripRawToDigi {
  
 public: // ----- public interface -----
  
  /** */
  SiStripRawToDigi( SiStripConnection& conns );
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
  inline Fed9U::Fed9UEvent* fed9UEvent();
  /** */
  inline Fed9U::Fed9UDescription* fed9UDescription();

  /** */
  void ZSMode(StripDigiCollection& digis, raw::FEDRawData& FEDBuffer, unsigned short fed_id);
  /** */
  void scopeMode(StripDigiCollection& digis, raw::FEDRawData& FEDBuffer, unsigned short fed_id);
  /** */
  void virginRaw(StripDigiCollection& digis, raw::FEDRawData& FEDBuffer, unsigned short fed_id);
  /** */
  void processedRaw(StripDigiCollection& digis, raw::FEDRawData& FEDBuffer, unsigned short fed_id);

  /** */
  inline int readoutOrder( int physical_order );
  /** */
  inline int physicalOrder( int readout_order ); 

  /** */
  void extractHeaderInfo() {;} //@@ needs implementing!
  /** */
  void debug();

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
  std::string readoutPath;
  /** */
  Fed9U::Fed9UDaqMode readoutMode;

  // some debug 
  vector<unsigned int> position;
  vector<unsigned int> landau;
  vector<unsigned short> fedids;
  unsigned long nFeds_;
  unsigned long nDets_;
  unsigned long nDigis_;

};

#endif // SiStripRawToDigi_H

// inline methods

inline Fed9U::Fed9UEvent* SiStripRawToDigi::fed9UEvent() {
  return fedEvent_;
}

inline Fed9U::Fed9UDescription* SiStripRawToDigi::fed9UDescription() {
  return description_;
}

inline void SiStripRawToDigi::fedReadoutPath(std::string rpath) {
  readoutPath = rpath;
}

inline void SiStripRawToDigi::fedReadoutMode(std::string rmode) {
  /* does nothing presently */
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



