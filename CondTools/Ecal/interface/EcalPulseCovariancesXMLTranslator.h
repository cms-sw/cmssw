/**
   Translates a EcalGainRatio record to XML and vice versa   
   \author Francesco RUBBO
   \version $Id: EcalPulseCovariancesXMLTranslator.h,v 0 2010/04/26 fay Exp $
   \date 26 Apr 2010
*/

#ifndef __EcalPulseCovariancesXMLTranslator_h_
#define __EcalPulseCovariancesXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"

#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>
#include <xercesc/dom/DOMNode.hpp>

//class EcalPulseCovariances;

class EcalPulseCovariancesXMLTranslator {

public:

  static int readXML (const std::string& filename, 
	              EcalCondHeader& header,
	              EcalPulseCovariances& record);

  static  int writeXML(const std::string& filename, 
		       const EcalCondHeader& header,
		       const EcalPulseCovariances& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalPulseCovariances& record);

};

#endif // __EcalPulseCovariancesXMLTranslator_h_
