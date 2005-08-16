#ifndef SiStripDigiToRaw_H
#define SiStripDigiToRaw_H

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CalibTracker/SiStripConnectivity/interface/SiStripConnection.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"

/**
   \class SiStripDigiToRaw
   \brief Short description here.
   \author M.Wingham, R.Bainbridge
   \version 0.1
   \date 09/08/05
   
   Long description here.
*/
class SiStripDigiToRaw {
  
 public:
  
  /** */
  SiStripDigiToRaw( SiStripConnection& conns );
  /** */
  ~SiStripDigiToRaw();
  
  /** */
  void createFedBuffers( StripDigiCollection& digis, 
			 raw::FEDRawDataCollection& fed_buffers );
  
 private:

  /** private default constructor */
  SiStripDigiToRaw() {;}

  /** */
  SiStripConnection connections_;
  /** */
  unsigned short verbosity_;


};

#endif // SiStripDigiToRaw_H
