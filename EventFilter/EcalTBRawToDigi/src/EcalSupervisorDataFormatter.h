#ifndef EcalSupervisorDataFormatter_H
#define EcalSupervisorDataFormatter_H
/** \class EcalSupervisorDataFormatter
 *
 *  $Id: EcalSupervisorDataFormatter.h,v 1.1 2006/07/21 12:36:25 meridian Exp $
 */

#include <TBDataFormats/EcalTBObjects/interface/EcalTBCollections.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

class FEDRawData;
class EcalSupervisorDataFormatter   {

 public:

  EcalSupervisorDataFormatter() ;
  virtual ~EcalSupervisorDataFormatter(){LogDebug("EcalTBRawToDigi") << "@SUB=EcalSupervisorDataFormatter" << "\n"; };

  //Method to be implemented
  void  interpretRawData( const FEDRawData & data, EcalTBEventHeader& tbEventHeader ) ;

 private:

  static const int nWordsPerEvent = 14;    

};
#endif
