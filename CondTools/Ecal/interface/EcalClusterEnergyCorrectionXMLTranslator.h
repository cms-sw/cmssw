/**
   Translates a EcalClusterEnergyCorrection record to XML
   and vice versa   

   \author 
   \version $Id: EcalClusterEnergyCorrectionXMLTranslator.h,v 1.0  $
   \date October 2011
*/

#ifndef __EcalClusterEnergyCorrectionXMLTranslator_h_
#define __EcalClusterEnergyCorrectionXMLTranslator_h_

#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include <xercesc/dom/DOMNode.hpp>
#include <string>
#include <vector>


class EcalClusterEnergyCorrectionXMLTranslator {

public:

  static int readXML  (const std::string& filename,
		       EcalCondHeader& header,
		       EcalFunParams& record);

  static int writeXML (const std::string& filename, 
		       const EcalCondHeader& header,
		       const EcalFunParams& record);
  
  // dump the CMSSW object container to XML
  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalFunParams& record);

};



#endif // __EcalClusterEnergyCorrectionXMLTranslator_h_
