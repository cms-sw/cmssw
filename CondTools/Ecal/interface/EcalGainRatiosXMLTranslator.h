/**
   Translates a EcalGainRatio record to XML
   and vice versa   
   \author Francesco RUBBO
   \version $Id: EcalGainRatiosXMLTranslator.h,v 1.1 2008/11/14 15:46:05 argiro Exp $
   \date 29 Jul 2008
*/

#ifndef __EcalGainRatiosXMLTranslator_h_
#define __EcalGainRatiosXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

class EcalGainRatiosXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalGainRatios& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalGainRatios& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalGainRatios& record);
};

#endif  // __EcalGainRatiosXMLTranslator_h_
