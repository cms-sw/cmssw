#ifndef CamacTBDataFormatter_H
#define CamacTBDataFormatter_H
/** \class CamacTBDataFormatter
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
class CamacTBDataFormatter   {

 public:

  CamacTBDataFormatter() {};
  virtual ~CamacTBDataFormatter(){LogDebug("EcalTBRawToDigi") << "@SUB=CamacTBDataFormatter" << "\n"; };

  //Method to be implemented
  void  interpretRawData( const FEDRawData & data, EcalTBEventHeader& tbEventHeader, EcalTBHodoscopeRawInfo& hodoRaw, EcalTBTDCRawInfo& tdcRawInfo ) {};
 private:

};
#endif
