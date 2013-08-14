#ifndef MatacqTBDataFormatter_H
#define MatacqTBDataFormatter_H
/** \class MatacqDataFormatter
 *
 *  $Id: MatacqDataFormatter.h,v 1.5 2007/10/20 10:58:01 franzoni Exp $
 */

#include <ostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include  "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

class MatacqTBRawEvent;
class FEDRawData;

class MatacqTBDataFormatter{
public:
  MatacqTBDataFormatter() {};
  virtual ~MatacqTBDataFormatter(){LogDebug("EcalTBRawToDigi") << "@SUB=MatacqTBDataFormatter" << "\n"; };
  
  /** Callback method for decoding raw data
   * @param data raw data
   * @param matacqDigiCollection [out] digi collection object to fill with
   * the decoded data
   */
  void  interpretRawData(const FEDRawData & data,
			 EcalMatacqDigiCollection& matacqDigiCollection);
  
private:
  void printData(std::ostream& out, const MatacqTBRawEvent& event) const;
};
#endif

