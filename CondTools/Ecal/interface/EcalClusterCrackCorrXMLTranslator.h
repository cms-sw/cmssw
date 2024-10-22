/**
   Translates a EcalFloatClusterCrackCorr record to XML and vice versa   

   \author 
   \version $Id: EcalClusterCrackCorrXMLTranslator.h,v 1.0  $
   \date October 2011
*/

#ifndef __EcalClusterCrackCorrXMLTranslator_h_
#define __EcalClusterCrackCorrXMLTranslator_h_

#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include <string>

class EcalClusterCrackCorrXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalFunParams& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalFunParams& record);

private:
  // dump the CMSSW object container to XML
  static std::string dumpXML(const EcalCondHeader& header, const EcalFunParams& record);
};

#endif  // __EcalClusterCrackCorrXMLTranslator_h_
