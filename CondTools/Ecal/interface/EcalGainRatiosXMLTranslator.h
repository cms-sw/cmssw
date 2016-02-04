/**
   Translates a EcalGainRatio record to XML
   and vice versa   
   \author Francesco RUBBO
   \version $Id: EcalGainRatiosXMLTranslator.h,v 1.2 2009/06/30 14:40:11 argiro Exp $
   \date 29 Jul 2008
*/

#ifndef __EcalGainRatiosXMLTranslator_h_
#define __EcalGainRatiosXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"

#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>
#include <xercesc/dom/DOMNode.hpp>

static const char CVSId__EcalGainRatiosXMLTranslator[] = 
"$Id: EcalGainRatiosXMLTranslator.h,v 1.2 2009/06/30 14:40:11 argiro Exp $";


//class EcalGainRatios;

class EcalGainRatiosXMLTranslator {

public:


  static int readXML (const std::string& filename, 
	              EcalCondHeader& header,
	              EcalGainRatios& record);

  static  int writeXML(const std::string& filename, 
		       const EcalCondHeader& header,
		       const EcalGainRatios& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalGainRatios& record);
  

};



#endif // __EcalGainRatiosXMLTranslator_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; scram b"
// End:
