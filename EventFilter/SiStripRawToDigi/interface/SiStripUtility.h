#ifndef SiStripUtility_H
#define SiStripUtility_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <string>

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

  /** Constructor that takes EventSetup as an argument, which is used
      to access the geometry service (which in turn defines the "size"
      of the cabling map). */
  SiStripUtility( const edm::EventSetup& );
  /** Destructor (presently does nothing). */
  ~SiStripUtility();
  
  /** Fills the cabling map (SiStripConnection, which is passed by
      reference) with "dummy" connections for "nDets_" detectors. */
  void siStripConnection( SiStripConnection& );
  /** Fills the StripDigiCollection (passed by reference) with random
      numbers of StripDigis with random positions and adc values. */
  int stripDigiCollection( StripDigiCollection& );
  /** Fills the FEDRawDataCollection (passed by reference) with
      FEDRawData objects that own FED buffers containing signal with
      random position and magnitude. Number of FED buffers is defined
      by the number of detectors (nDets_). */
  void fedRawDataCollection( FEDRawDataCollection& );

  /** Sets the FED readout mode (ZS, VR, PR, SM) and thus the type of
      FED buffer to be created. */
  inline void fedReadoutMode( std::string readout_mode );

  /** set verbosity */ 
  inline void verbose( bool );

private:

  /** Private default constructor. */
  SiStripUtility();
  /** Number of detectors. */
  int nDets_;
  /** The FED readout mode (ZS, VR, PR, SM). */
  std::string fedReadoutMode_;
  /** verbosity switch */
  bool verbose_;

};

void SiStripUtility::fedReadoutMode( std::string readout_mode ) {
  fedReadoutMode_ = readout_mode;
}

void SiStripUtility::verbose( bool verbose ) { 
  verbose_ = verbose; 
}

#endif // SiStripUtility_H

