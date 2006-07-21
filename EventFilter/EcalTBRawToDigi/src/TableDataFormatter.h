#ifndef TableDataFormatter_H
#define TableDataFormatter_H
/** \class TableDataFormatter
 *
 *  $Id: $
 */

#include <TBDataFormats/EcalTBObjects/interface/EcalTBCollections.h>

#include <vector> 
#include <map>
#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

class FEDRawData;
class TableDataFormatter   {

 public:

  TableDataFormatter() {};
  virtual ~TableDataFormatter(){LogDebug("EcalTBRawToDigi") << "@SUB=TableDataFormatter" << "\n"; };

  //Method to be implemented
  void  interpretRawData( const FEDRawData & data, EcalTBEventHeader& tbEventHeader) {};
 private:

};
#endif
