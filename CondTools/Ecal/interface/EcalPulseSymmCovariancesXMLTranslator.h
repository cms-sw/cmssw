/**
   Translates a EcalGainRatio record to XML and vice versa   
   \author Francesco RUBBO
   \version $Id: EcalPulseSymmCovariancesXMLTranslator.h,v 0 2010/04/26 fay Exp $
   \date 26 Apr 2010
*/

#ifndef __EcalPulseSymmCovariancesXMLTranslator_h_
#define __EcalPulseSymmCovariancesXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalPulseSymmCovariances.h"

#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>
#include <xercesc/dom/DOMNode.hpp>

//class EcalPulseSymmCovariances;

class EcalPulseSymmCovariancesXMLTranslator {

public:

  static int readXML (const std::string& filename, 
	              EcalCondHeader& header,
	              EcalPulseSymmCovariances& record);

  static  int writeXML(const std::string& filename, 
		       const EcalCondHeader& header,
		       const EcalPulseSymmCovariances& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalPulseSymmCovariances& record);

};

#endif // __EcalPulseSymmCovariancesXMLTranslator_h_
