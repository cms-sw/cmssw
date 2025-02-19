#ifndef MatacqDataFormatter_H
#define MatacqDataFormatter_H

#include <ostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "EventFilter/EcalRawToDigi/src/MatacqDataFormatter.h"

class MatacqRawEvent;
class FEDRawData;

/** Class to interpret ECAL MATACQ raw data and produce the MATACQ digis.
 * This class is used by the MatacqProducer module.
 *  @author: Ph. Gras (CEA/Saclay)
 *  $Id: MatacqDataFormatter.h,v 1.1 2009/02/25 14:44:25 pgras Exp $
 */
class MatacqDataFormatter{
public:
  MatacqDataFormatter() {};
  
  /** Callback method for decoding raw data
   * @param data raw data
   * @param matacqDigiCollection [out] digi collection object to fill with
   * the decoded data
   */
  void interpretRawData(const FEDRawData & data,
			EcalMatacqDigiCollection& matacqDigiCollection);
  
  /** Callback method for decoding raw data
   * @param data raw data
   * @param matacqDigiCollection [out] digi collection object to fill with
   * the decoded data
   */
  void interpretRawData(const MatacqRawEvent& data,
			EcalMatacqDigiCollection& matacqDigiCollection);
  
private:
  void printData(std::ostream& out, const MatacqRawEvent& event) const;
};
#endif

