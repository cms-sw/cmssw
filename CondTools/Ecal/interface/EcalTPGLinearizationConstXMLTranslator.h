/**
   Translates a EcalTPGLinearizationConst record to XML
   \version $Id: EcalTPGLinearizationConstXMLTranslator.h,v 1.0 2012/05/01 13:48:05 Exp $
   \date 01 May 2012
*/

#ifndef __EcalTPGLinearizationConstXMLTranslator_h_
#define __EcalTPGLinearizationConstXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

class EcalTPGLinearizationConstXMLTranslator {
public:
  static int writeXML(const std::string& filename,
                      const EcalCondHeader& header,
                      const EcalTPGLinearizationConst& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalTPGLinearizationConst& record);
};

#endif  // __EcalTPGLinearizationConstXMLTranslator_h_
