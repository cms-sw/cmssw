/**
   Translates a EcalAlignmentConstant record to XML
   and vice versa   

   \author Jean Fay
   \version $Id: EcalAlignmentXMLTranslator.h,v 1.1 2010/10/15 17:13:36 fay Exp $
   \date 14 Sept 2010
*/

#ifndef __EcalAlignmentXMLTranslator_h_
#define __EcalAlignmentXMLTranslator_h_

#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

class Alignments;

class EcalAlignmentXMLTranslator {
public:
  static int writeXML(const std::string& filename, const EcalCondHeader& header, const Alignments& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const Alignments& record);
};

#endif  // __EcalAlignmentXMLTranslator_h_
