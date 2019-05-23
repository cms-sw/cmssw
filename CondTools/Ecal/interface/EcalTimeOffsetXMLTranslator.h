/**
   Translates an EcalTimeOffsetConstant record to XML
   and vice versa   

   \author Seth Cooper, University of Minnesota
   \version $Id: $
   \date 21 Mar 2011
*/

#ifndef __EcalTimeOffsetXMLTranslator_h_
#define __EcalTimeOffsetXMLTranslator_h_

#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

class EcalTimeOffsetConstant;

class EcalTimeOffsetXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalTimeOffsetConstant& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalTimeOffsetConstant& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalTimeOffsetConstant& record);
};

#endif  // __EcalTimeOffsetXMLTranslator_h_
