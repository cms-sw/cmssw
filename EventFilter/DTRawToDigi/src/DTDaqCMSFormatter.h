#ifndef DTDaqCMSFormatter_H
#define DTDaqCMSFormatter_H
/** \class DTDaqCMSFormatter
 *  
 *  This class allows to transform the DT raw data of a given
 *  Readout Unit into OO digi, which are understood by the
 *  ORCA reconstruction code (method interpretRawData). 
 *  It also accomplishes the inverse transformation, when one wants 
 *  to create raw data out of simulated detector digis written on a 
 *  DataBase (method formatData).
 *
 *  $Date: 2005/07/06 15:52:01 $
 *  $Revision: 1.1 $
 *  \author G. Bruno - CERN, EP Division
 */

// #include "CommonDet/DaqDetInterface/interface/DaqFEDFormatterSingleDigi.h"
// #include "Muon/MBDetector/interface/MuBarBaseReadout.h"
#include <string>
//#include <DataFormats/Digis/interface/DTDigiCollection.h>

namespace raw {class FEDRawData;}
class 	DTDigiCollection;

class DTDaqCMSFormatter {

 public:

  void interpretRawData(const raw::FEDRawData & data, 
			DTDigiCollection& digicollection);

  //  DaqFEDRawData *  formatData(FrontEndDriver * fed);


  // From DaqFEDFormatter...
  inline void checkMemory(int totsize, int alreadyfilled, int requested){
    if (alreadyfilled+requested > totsize) throw std::string("DaqFEDFormatter::checkMemory() - ERROR: the requested memory exceeds the reserved one");

  }

};

#endif
