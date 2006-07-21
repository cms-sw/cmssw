#ifndef EcalSupervisorDataFormatter_H
#define EcalSupervisorDataFormatter_H
/** \class EcalSupervisorDataFormatter
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
class EcalSupervisorDataFormatter   {

 public:

  EcalSupervisorDataFormatter() {};
  virtual ~EcalSupervisorDataFormatter(){LogDebug("EcalTBRawToDigi") << "@SUB=EcalSupervisorDataFormatter" << "\n"; };

  //Method to be implemented
  void  interpretRawData( const FEDRawData & data, EcalTBEventHeader& tbEventHeader ) {};
 private:

};
#endif
