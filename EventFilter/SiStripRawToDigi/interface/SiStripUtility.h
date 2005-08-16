#ifndef SiStripUtility_H
#define SiStripUtility_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

//class EventSetup;
class SiStripConnection;

/**
   \class SiStripUtility 
   \brief Short description here
   \author R.Bainbridge
   \version 0.1
   \date 09/08/05
   
   Long description here.
*/
class SiStripUtility {
  
public:

  /** */
  SiStripUtility( const edm::EventSetup& );
  /** */
  ~SiStripUtility();
  
  /** */
  void siStripConnection( SiStripConnection& );
  /** */
  void stripDigiCollection( StripDigiCollection& );
  /** */
  void fedRawDataCollection( raw::FEDRawDataCollection& );

private:

  /** private constructor */
  SiStripUtility();
  /** */
  int nDets_;
  
};

#endif // SiStripUtility_H

