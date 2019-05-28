/**
   Translates a EcalClusterLocalContCorr record to XML
   and vice versa   
*/

#ifndef __EcalClusterLocalContCorrXMLTranslator_h_
#define __EcalClusterLocalContCorrXMLTranslator_h_

#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include <string>

class EcalClusterLocalContCorrXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalFunParams& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalFunParams& record);

private:
  // dump the CMSSW object container to XML
  static std::string dumpXML(const EcalCondHeader& header, const EcalFunParams& record);
};

#endif  // __EcalClusterLocalContCorrXMLTranslator_h_
