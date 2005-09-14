#ifndef SiStripDigiToRaw_H
#define SiStripDigiToRaw_H

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
   \class SiStripDigiToRaw

   \brief Input: StripDigiCollection. Output: FEDRawDataCollection.
   \author M.Wingham, R.Bainbridge
   \version 0.1
   \date 05/09/05
   
   Input: StripDigiCollection. 
   Output: FEDRawDataCollection.
*/
class SiStripDigiToRaw {
  
 public: // ----- public interface -----
  
  /** */
  SiStripDigiToRaw( SiStripConnection& conns,
		    unsigned short verbosity = 0 );
  /** */
  ~SiStripDigiToRaw();
  
  /** Takes a StripDigiCollection as input and creates a
      FEDRawDataCollection. */
  void createFedBuffers( StripDigiCollection& digis, 
			 raw::FEDRawDataCollection& fed_buffers );

  /** */
  inline void fedReadoutPath( std::string rpath );
  /** */
  inline void fedReadoutMode( std::string rmode );

 private: // ----- private methods -----

  /** private default constructor */
  SiStripDigiToRaw() {;}

 private: // ----- private data members -----

  /** */
  SiStripConnection connections_;
  /** */
  unsigned short verbosity_;

  /** */
  std::string readoutPath_;
  /** */
  std::string readoutMode_;


  /** FED identifiers. */
  vector<unsigned short> fedids_;

  // some debug counters
  vector<unsigned int> position_;
  vector<unsigned int> landau_;
  unsigned long nFeds_;
  unsigned long nDets_;
  unsigned long nDigis_;

};

#endif // SiStripDigiToRaw_H

// inline methods 

inline void SiStripDigiToRaw::fedReadoutPath( std::string rpath ) { 
  readoutPath_ = rpath;
}

inline void SiStripDigiToRaw::fedReadoutMode( std::string rmode ) { 
  readoutMode_ = rmode;
}
