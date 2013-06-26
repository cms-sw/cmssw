/**
   Translates a EcalClusterEnergyCorrection record to XML
   and vice versa   

   \author 
   \version $Id: EcalClusterEnergyCorrectionObjectSpecificXMLTranslator.h,v 1.1 2011/11/10 17:52:22 fay Exp $
   \date November 2011
*/

#ifndef __EcalClusterEnergyCorrectionObjectSpecificXMLTranslator_h_
#define __EcalClusterEnergyCorrectionObjectSpecificXMLTranslator_h_

#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include <xercesc/dom/DOMNode.hpp>
#include <string>
#include <vector>


class EcalClusterEnergyCorrectionObjectSpecificXMLTranslator {

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
