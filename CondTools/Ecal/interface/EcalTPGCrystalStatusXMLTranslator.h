/**
   Translates a Ecal record to XML and vice versa   
   \version $Id: EcalTPGCrystalStatusXMLTranslator.h,v 1.1 2011/06/22 12:49:18 fay Exp $
   \date 4 Apr 2011
*/

#ifndef __EcalTPGCrystalStatusXMLTranslator_h_
#define __EcalTPGCrystalStatusXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"

#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>
#include <xercesc/dom/DOMNode.hpp>

static const char CVSId__EcalTPGCrystalStatusXMLTranslator[] = 
"$Id: EcalTPGCrystalStatusXMLTranslator.h,v 1.1 2011/06/22 12:49:18 fay Exp $";


//class EcalTPGCrystalStatus;

class EcalTPGCrystalStatusXMLTranslator {

public:

  static  int writeXML(const std::string& filename, 
		       const EcalCondHeader& header,
		       const EcalTPGCrystalStatus& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalTPGCrystalStatus& record);

  static void plot(std::string, const EcalTPGCrystalStatus& record);
};

#endif // __EcalTPGCrystalStatusXMLTranslator_h_
