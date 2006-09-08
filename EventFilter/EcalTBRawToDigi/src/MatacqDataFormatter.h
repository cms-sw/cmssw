#ifndef MatacqDataFormatter_H
#define MatacqDataFormatter_H
/** \class MatacqDataFormatter
 *
 *  $Id: MatacqDataFormatter.h,v 1.1 2006/07/21 12:36:25 meridian Exp $
 */

#include <ostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class MatacqRawEvent;
class FEDRawData;
class EcalMatacqDigi;

class MatacqDataFormatter{
public:
  MatacqDataFormatter() {};
  virtual ~MatacqDataFormatter(){LogDebug("EcalTBRawToDigi") << "@SUB=MatacqDataFormatter" << "\n"; };
  
  //Method to be implemented
  void  interpretRawData(const FEDRawData & data, EcalMatacqDigi& matacqDigi);
  
private:
  void printData(std::ostream& out, const MatacqRawEvent& event) const;
};
#endif

