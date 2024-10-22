/**
   Translates a EcalGainRatio record to XML and vice versa   
   \author Francesco RUBBO
   \version $Id: EcalPedestalsXMLTranslator.h,v 0 2010/04/26 fay Exp $
   \date 26 Apr 2010
*/

#ifndef __EcalPedestalsXMLTranslator_h_
#define __EcalPedestalsXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

class EcalPedestalsXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalPedestals& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalPedestals& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalPedestals& record);
};

#endif  // __EcalPedestalsXMLTranslator_h_
