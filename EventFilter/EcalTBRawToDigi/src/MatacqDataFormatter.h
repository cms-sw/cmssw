#ifndef MatacqTBDataFormatter_H
#define MatacqTBDataFormatter_H
/** \class MatacqDataFormatter
 *
 *  $Id: MatacqTBDataFormatter.h,v 1.4 2006/09/21 12:22:36 pgras Exp $
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

