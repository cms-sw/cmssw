#ifndef SiStripRawToDigi_H
#define SiStripRawToDigi_H

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CalibTracker/SiStripConnectivity/interface/SiStripConnection.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"

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
  
 public:
  
  /** */
  SiStripRawToDigi( SiStripConnection& conns );
  /** */
  ~SiStripRawToDigi();
  
  /** Takes a FEDRawDataCollection as input and creates a
      StripDigiCollection. */
  void createDigis( raw::FEDRawDataCollection& fed_buffers,
		    StripDigiCollection& digis );
  
 private:

  /** private default constructor */
  SiStripRawToDigi() {;}

  /** */
  SiStripConnection connections_;
  /** */
  unsigned short verbosity_;


};

#endif // SiStripRawToDigi_H
