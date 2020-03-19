/**
   Translates a EcalLinearCorrections record to XML
   and vice versa   

   \author Stefano ARGIRO
   \version $Id: EcalLinearCorrectionsXMLTranslator.h,v 1.1 2012/11/21 17:01:39 fra Exp $
   \date 20 Jun 2008
*/
#ifndef __EcalLinearCorrectionsXMLTranslator_h_
#define __EcalLinearCorrectionsXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

class EcalLinearCorrectionsXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalLinearCorrections& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalLinearCorrections& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalLinearCorrections& record);
};

#endif  // __EcalLinearCorrectionsXMLTranslator_h_
