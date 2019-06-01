#ifndef EcalSupervisorTBDataFormatter_H
#define EcalSupervisorTBDataFormatter_H
/** \class EcalSupervisorTBDataFormatter
 *
 */

#include <TBDataFormats/EcalTBObjects/interface/EcalTBCollections.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class FEDRawData;
class EcalSupervisorTBDataFormatter {
public:
  EcalSupervisorTBDataFormatter();
  virtual ~EcalSupervisorTBDataFormatter() {
    LogDebug("EcalTBRawToDigi") << "@SUB=EcalSupervisorTBDataFormatter"
                                << "\n";
  };

  //Method to be implemented
  void interpretRawData(const FEDRawData& data, EcalTBEventHeader& tbEventHeader);

private:
  static const int nWordsPerEvent = 14;
};
#endif
