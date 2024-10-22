/**
   Translates a Ecal record to XML
   \version $Id: EcalDAQStripStatusXMLTranslator.h,v 1.1 2011/06/14 fay Exp $
   \date 14 Jun 2011
*/

#ifndef __EcalTPGStripStatusXMLTranslator_h_
#define __EcalTPGStripStatusXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

class EcalTPGStripStatusXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalTPGStripStatus& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalTPGStripStatus& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalTPGStripStatus& record);
};

#endif  // __EcalTPGStripStatusXMLTranslator_h_
