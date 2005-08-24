#ifndef SiStripDigiToRaw_H
#define SiStripDigiToRaw_H

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CalibTracker/SiStripConnectivity/interface/SiStripConnection.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"

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
  
 public:
  
  /** */
  SiStripDigiToRaw( SiStripConnection& conns );
  /** */
  ~SiStripDigiToRaw();
  
  /** Takes a StripDigiCollection as input and creates a
      FEDRawDataCollection. */
  void createFedBuffers( StripDigiCollection& digis, 
			 raw::FEDRawDataCollection& fed_buffers );

  void fedReadoutMode( string ) {;}
  void fedReadoutPath( string ) {;}
  
 private:

  /** private default constructor */
  SiStripDigiToRaw() {;}

  /** */
  SiStripConnection connections_;
  /** */
  unsigned short verbosity_;


};

#endif // SiStripDigiToRaw_H
