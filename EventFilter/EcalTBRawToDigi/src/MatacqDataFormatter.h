#ifndef MatacqDataFormatter_H
#define MatacqDataFormatter_H
/** \class MatacqDataFormatter
 *
 *  $Id: MatacqDataFormatter.h,v 1.3 2006/09/08 13:23:38 pgras Exp $
 */

#include <ostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include  "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

class MatacqRawEvent;
class FEDRawData;

class MatacqDataFormatter{
public:
  MatacqDataFormatter() {};
  virtual ~MatacqDataFormatter(){LogDebug("EcalTBRawToDigi") << "@SUB=MatacqDataFormatter" << "\n"; };
  
  /** Callback method for decoding raw data
   * @param data raw data
   * @param matacqDigiCollection [out] digi collection object to fill with
   * the decoded data
   */
  void  interpretRawData(const FEDRawData & data,
			 EcalMatacqDigiCollection& matacqDigiCollection);
  
private:
  void printData(std::ostream& out, const MatacqRawEvent& event) const;
};
#endif

