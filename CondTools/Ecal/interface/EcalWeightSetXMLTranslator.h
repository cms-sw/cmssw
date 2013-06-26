/**
   Translates a EcalWeightSet record to XML
   and vice versa   

   \author Stefano ARGIRO
   \version $Id: EcalWeightSetXMLTranslator.h,v 1.2 2009/06/30 14:40:11 argiro Exp $
   \date 20 Jun 2008
*/

#ifndef __EcalWeightSetXMLTranslator_h_
#define __EcalWeightSetXMLTranslator_h_


#include "CondTools/Ecal/interface/XercesString.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <string>


static const char CVSId__EcalWeightSetXMLTranslator[] = 
"$Id: EcalWeightSetXMLTranslator.h,v 1.2 2009/06/30 14:40:11 argiro Exp $";



class EcalWeightSetXMLTranslator {

public:
  
  EcalWeightSetXMLTranslator(){};

  static int readXML  (const std::string& filename, 
		       EcalCondHeader& header,
		       EcalWeightSet& record);

  static int writeXML (const std::string& filename,
		       const EcalCondHeader& header, 
		       const EcalWeightSet& record);
  
  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalWeightSet&  record);  


private:
  
  static void write10x10(xercesc::DOMElement* node,const EcalWeightSet& record);
  static void write3x10(xercesc::DOMElement* node,const EcalWeightSet& record);
 
};



#endif // __EcalWeightSetXMLTranslator_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; scram b"
// End:
