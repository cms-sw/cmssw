/**
   Translates a Ecal record to XML and vice versa   
   \version $Id: EcalTPGTowerStatusXMLTranslator.h,v 1.1 2011/06/22 12:52:52 fay Exp $
   \date 4 Apr 2011
*/

#ifndef __EcalTPGTowerStatusXMLTranslator_h_
#define __EcalTPGTowerStatusXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"

#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>
#include <xercesc/dom/DOMNode.hpp>

static const char CVSId__EcalTPGTowerStatusXMLTranslator[] = 
"$Id: EcalTPGTowerStatusXMLTranslator.h,v 1.1 2011/06/22 12:52:52 fay Exp $";


//class EcalTPGTowerStatus;

class EcalTPGTowerStatusXMLTranslator {

public:

  static int readXML (const std::string& filename, 
	              EcalCondHeader& header,
	              EcalTPGTowerStatus& record);

  static  int writeXML(const std::string& filename, 
		       const EcalCondHeader& header,
		       const EcalTPGTowerStatus& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalTPGTowerStatus& record);

  static void plot(std::string, const EcalTPGTowerStatus& record);
};

#endif // __EcalTPGTowerStatusXMLTranslator_h_
