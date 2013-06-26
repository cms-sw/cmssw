/**
   Translates a EcalClusterLocalContCorr record to XML
   and vice versa   

   \author 
   \version $Id: EcalClusterLocalContCorrXMLTranslator.h,v 1.1 2011/11/10 17:54:20 fay Exp $
   \date October 2011
*/

#ifndef __EcalClusterLocalContCorrXMLTranslator_h_
#define __EcalClusterLocalContCorrXMLTranslator_h_

#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include <xercesc/dom/DOMNode.hpp>
#include <string>
#include <vector>


class EcalClusterLocalContCorrXMLTranslator {

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



#endif // __EcalClusterLocalContCorrXMLTranslator_h_
