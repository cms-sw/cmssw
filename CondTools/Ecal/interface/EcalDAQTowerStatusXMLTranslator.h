/**
   Translates a EcalGainRatio record to XML and vice versa   
   \author Francesco RUBBO
   \version $Id: EcalDAQTowerStatusXMLTranslator.h,v 1.1 2010/07/29 16:43:48 fay Exp $
   \date 26 Apr 2010
*/

#ifndef __EcalDAQTowerStatusXMLTranslator_h_
#define __EcalDAQTowerStatusXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"

#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>
#include <xercesc/dom/DOMNode.hpp>

static const char CVSId__EcalDAQTowerStatusXMLTranslator[] = 
"$Id: EcalDAQTowerStatusXMLTranslator.h,v 1.1 2010/07/29 16:43:48 fay Exp $";


//class EcalDAQTowerStatus;

class EcalDAQTowerStatusXMLTranslator {

public:

  static int readXML (const std::string& filename, 
	              EcalCondHeader& header,
	              EcalDAQTowerStatus& record);

  static  int writeXML(const std::string& filename, 
		       const EcalCondHeader& header,
		       const EcalDAQTowerStatus& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalDAQTowerStatus& record);

  static void plot(std::string, const EcalDAQTowerStatus& record);
};

#endif // __EcalDAQTowerStatusXMLTranslator_h_
