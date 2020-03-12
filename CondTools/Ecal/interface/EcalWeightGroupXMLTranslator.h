/**
   Translates a EcalWeightGroup record to XML
   and vice versa   

   \author Stefano ARGIRO
   \version $Id: EcalWeightGroupXMLTranslator.h,v 1.2 2009/06/30 14:40:11 argiro Exp $
   \date 20 Jun 2008
*/

#ifndef _EcalWeightGroupXMLTranslator_h_
#define _EcalWeightGroupXMLTranslator_h_

#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include <string>

class EcalWeightGroupXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalWeightXtalGroups& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalWeightXtalGroups& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalWeightXtalGroups& record);
};

#endif  // __EcalWeightGroupXMLTranslator_h_
