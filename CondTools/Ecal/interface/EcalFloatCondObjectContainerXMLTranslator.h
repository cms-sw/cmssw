/**
   Translates a EcalFloatCondObjectContainer record to XML
   and vice versa   

   \author Stefano ARGIRO
   \version $Id: EcalFloatCondObjectContainerXMLTranslator.h,v 1.4 2011/05/04 12:38:10 argiro Exp $
   \date 20 Jun 2008
*/

#ifndef __EcalFloatCondObjectContainterXMLTranslator_h_
#define __EcalFloatCondObjectContainterXMLTranslator_h_

#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include <xercesc/dom/DOMNode.hpp>
#include <string>
#include <vector>


class EcalFloatCondObjectContainerXMLTranslator {

public:
  

  static int readXML  (const std::string& filename,
		       EcalCondHeader& header,
		       EcalFloatCondObjectContainer& record);

  static std::vector<float>  barrelfromXML(const std::string& filename);
		      
  static std::vector<float>  endcapfromXML(const std::string& filename);

  static int writeXML (const std::string& filename, 
		       const EcalCondHeader& header,
		       const EcalFloatCondObjectContainer& record);
  
  // dump the CMSSW object container to XML
  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalFloatCondObjectContainer& record);

  // dump the two flat arrays (hashed-indexed as in EBDetId, EEDetId) to XML 
  static std::string dumpXML(const EcalCondHeader& header,
			     const std::vector<float>& eb,
			     const std::vector<float>& ee);

};



#endif // __EcalFloatCondObjectContainerXMLTranslator_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; scram b"
// End:
