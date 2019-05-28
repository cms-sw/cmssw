/**
   Translates a EcalWeightSet record to XML
   and vice versa   

   \author Stefano ARGIRO
   \version $Id: EcalWeightSetXMLTranslator.h,v 1.1 2008/11/14 15:46:05 argiro Exp $
   \date 20 Jun 2008
*/

#ifndef __EcalWeightSetXMLTranslator_h_
#define __EcalWeightSetXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <xercesc/dom/DOMElement.hpp>
#include <string>

class EcalWeightSetXMLTranslator {
public:
  EcalWeightSetXMLTranslator(){};

  static int readXML(const std::string& filename, EcalCondHeader& header, EcalWeightSet& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalWeightSet& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalWeightSet& record);

  static void write10x10(xercesc::DOMElement* node, const EcalWeightSet& record);
  static void write3x10(xercesc::DOMElement* node, const EcalWeightSet& record);
};

#endif  // __EcalWeightSetXMLTranslator_h_
