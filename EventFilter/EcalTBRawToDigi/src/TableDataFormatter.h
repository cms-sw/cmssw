#ifndef TableDataFormatter_H
#define TableDataFormatter_H
/** \class TableDataFormatter
 *
 *  $Id: TableDataFormatter.h,v 1.5 2007/04/12 08:36:47 franzoni Exp $
 */

#include <TBDataFormats/EcalTBObjects/interface/EcalTBCollections.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



class FEDRawData;
class TableDataFormatter   {

 public:

  TableDataFormatter() ;
  virtual ~TableDataFormatter(){LogDebug("EcalTBRawToDigi") << "@SUB=TableDataFormatter" << "\n"; };

  //Method to be implemented
  void  interpretRawData( const FEDRawData & data, EcalTBEventHeader& tbEventHeader);
 private:

 static const int nWordsPerEvent =10;    // Number of fibers per hodoscope plane   
};
#endif
