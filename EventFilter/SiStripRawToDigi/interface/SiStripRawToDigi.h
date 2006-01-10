#ifndef EventFilter_SiStripRawToDigi_H
#define EventFilter_SiStripRawToDigi_H

#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include "Fed9UUtils.hh"

class SiStripReadoutCabling;
class FEDRawDataCollection;
class StripDigiCollection;

/**
   \class SiStripRawToDigi
   \brief Takes a FEDRawDataCollection as input and creates a
      StripDigiCollection.
   \author M.Wingham, R.Bainbridge
*/
class SiStripRawToDigi {
  
 public: // ----- public interface -----
  
  SiStripRawToDigi( unsigned short verbosity );
  ~SiStripRawToDigi();
  
  /** Takes a FEDRawDataCollection as input and creates a
      StripDigiCollection. */
  void createDigis( edm::ESHandle<SiStripReadoutCabling>& cabling,
		    edm::Handle<FEDRawDataCollection>& buffers,
		    std::auto_ptr<StripDigiCollection>& digis ); //@@ <- is this good?
  
  inline void fedReadoutPath( std::string readout_path );
  inline void fedReadoutMode( std::string readout_mode );
  
 private: // ----- private methods -----
  
  /** */
  void zeroSuppr( unsigned short fed_id,
		  edm::ESHandle<SiStripReadoutCabling>& cabling,
		  std::auto_ptr<StripDigiCollection>& digis );
  /** */
  void rawModes( unsigned short fed_id,
		 edm::ESHandle<SiStripReadoutCabling>& cabling,
		 std::auto_ptr<StripDigiCollection>& digis );
  /** */
  void scopeMode( unsigned short fed_id,
		  edm::ESHandle<SiStripReadoutCabling>& cabling,
		  std::auto_ptr<StripDigiCollection>& digis );
  
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
  unsigned short verbosity_;

  /** */
  Fed9U::Fed9UDescription* fedDescription_;
  /** */
  Fed9U::Fed9UEvent* fedEvent_;

  /** */
  std::string readoutPath_;
  /** */
  std::string readoutMode_; // unsigned short readoutMode_;

  // some debug counters
  std::vector<unsigned int> position_;
  std::vector<unsigned int> landau_;
  unsigned long nFeds_;
  unsigned long nDets_;
  unsigned long nDigis_;

};

#endif // EventFilter_SiStripRawToDigi_H

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



