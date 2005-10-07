#ifndef DTRawToDigi_DTDaqCMSFormatter_h
#define DTRawToDigi_DTDaqCMSFormatter_h
/** \class DTDaqCMSFormatter
 *  
 *  This class allows to transform the DT raw data of a given
 *  Readout Unit into OO digi, which are understood by the
 *  ORCA reconstruction code (method interpretRawData). 
 *  It also accomplishes the inverse transformation, when one wants 
 *  to create raw data out of simulated detector digis written on a 
 *  DataBase (method formatData).
 *
 *  $Date: 2005/08/23 09:31:36 $
 *  $Revision: 1.2 $
 *  \author G. Bruno - CERN, EP Division
 */

#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <string>

class FEDRawData;


class DTDaqCMSFormatter {

 public:

  void interpretRawData(const FEDRawData & data, 
			DTDigiCollection& digicollection);

  //  DaqFEDRawData *  formatData(FrontEndDriver * fed);


  // From DaqFEDFormatter...
  inline void checkMemory(int totsize, int alreadyfilled, int requested){
    if (alreadyfilled+requested > totsize) throw std::string("DaqFEDFormatter::checkMemory() - ERROR: the requested memory exceeds the reserved one");

  }

};

#endif
