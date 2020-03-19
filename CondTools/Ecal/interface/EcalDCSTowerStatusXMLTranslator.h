/**
   Translates a EcalGainRatio record to XML and vice versa   
   \author Francesco RUBBO
   \version $Id: EcalDCSTowerStatusXMLTranslator.h,v 0 2010/06/3 fay Exp $
   \date 3 Jun 2010
*/

#ifndef __EcalDCSTowerStatusXMLTranslator_h_
#define __EcalDCSTowerStatusXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

class EcalDCSTowerStatusXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalDCSTowerStatus& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalDCSTowerStatus& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalDCSTowerStatus& record);

  static void plot(std::string, const EcalDCSTowerStatus& record);
};

#endif  // __EcalDCSTowerStatusXMLTranslator_h_
