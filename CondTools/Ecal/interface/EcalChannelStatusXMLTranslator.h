/**
   Translates a EcalChannelStatus record to XML
   and vice versa   
   \author Francesco RUBBO
   \version $Id: EcalChannelStatusXMLTranslator.h,v 1.1 2008/11/14 15:46:05 argiro Exp $
   \date 29 Jul 2008
*/

#ifndef __EcalChannelStatusXMLTranslator_h_
#define __EcalChannelStatusXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

class EcalChannelStatusXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalChannelStatus& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalChannelStatus& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalChannelStatus& record);
};

#endif  // __EcalChannelStatusXMLTranslator_h_
