#ifndef MatacqDataFormatter_H
#define MatacqDataFormatter_H
/** \class MatacqDataFormatter
 *
 *  $Id: $
 */

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <vector> 
#include <map>
#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

class FEDRawData;
class MatacqDataFormatter   {

 public:

  MatacqDataFormatter() {};
  virtual ~MatacqDataFormatter(){LogDebug("EcalTBRawToDigi") << "@SUB=MatacqDataFormatter" << "\n"; };

  //Method to be implemented
  void  interpretRawData( const FEDRawData & data, EcalMatacqDigi& matacqDigi ) {};
 private:

};
#endif
