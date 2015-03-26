/**
   Translates a EcalGainRatio record to XML and vice versa   
   \author Francesco RUBBO
   \version $Id: EcalPulseShapesXMLTranslator.h,v 0 2010/04/26 fay Exp $
   \date 26 Apr 2010
*/

#ifndef __EcalPulseShapesXMLTranslator_h_
#define __EcalPulseShapesXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"

#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>
#include <xercesc/dom/DOMNode.hpp>

//class EcalPulseShapes;

class EcalPulseShapesXMLTranslator {

public:

  static int readXML (const std::string& filename, 
	              EcalCondHeader& header,
	              EcalPulseShapes& record);

  static  int writeXML(const std::string& filename, 
		       const EcalCondHeader& header,
		       const EcalPulseShapes& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalPulseShapes& record);

};

#endif // __EcalPulseShapesXMLTranslator_h_
