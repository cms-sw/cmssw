#ifndef SiStripDigiToRaw_H
#define SiStripDigiToRaw_H

#include "CalibTracker/SiStripConnectivity/interface/SiStripConnection.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
//
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
//
#include "Fed9UUtils.hh"

/**
   \class SiStripDigiToRaw

   \brief Takes a StripDigiCollection as input and creates a
   FEDRawDataCollection.
   \author M.Wingham, R.Bainbridge
   \version 0.1
   \date 09/08/05
   
   Takes a StripDigiCollection as input and creates a
   FEDRawDataCollection.

*/
class SiStripDigiToRaw {
  
 public: // ----- public interface -----
  
  /** */
  SiStripDigiToRaw( SiStripConnection& conns );
  /** */
  ~SiStripDigiToRaw();
  
  /** Takes a StripDigiCollection as input and creates a
      FEDRawDataCollection. */
  void createFedBuffers( StripDigiCollection& digis, 
			 raw::FEDRawDataCollection& fed_buffers );

  /** */
  inline void fedReadoutPath(std::string rpath);
  /** */
  inline void fedReadoutMode(std::string rmode);
  
 private: // ----- private methods -----

  /** private default constructor */
  SiStripDigiToRaw() {;}

 private: // ----- private data members -----

  /** */
  SiStripConnection connections_;
  /** */
  unsigned short verbosity_;

  /** */
  std::string readoutPath;
  /** */
  std::string readoutMode;

  /** */
  vector<unsigned short> fedids;

};

#endif // SiStripDigiToRaw_H

// inline methods 

inline void SiStripDigiToRaw::fedReadoutPath(std::string rpath) { 
  readoutPath = rpath;
}

inline void SiStripDigiToRaw::fedReadoutMode(std::string rmode) { 
  readoutMode = rmode;
}
