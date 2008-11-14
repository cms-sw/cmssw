/**
   Translates a EcalWeightSet record to XML
   and vice versa   

   \author Stefano ARGIRO
   \version $Id: EcalWeightSetXMLTranslator.h,v 1.1 2008/11/06 08:36:18 argiro Exp $
   \date 20 Jun 2008
*/

#ifndef __EcalWeightSetXMLTranslator_h_
#define __EcalWeightSetXMLTranslator_h_


#include "CondTools/Ecal/interface/XercesString.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <string>


static const char CVSId__EcalWeightSetXMLTranslator[] = 
"$Id: EcalWeightSetXMLTranslator.h,v 1.1 2008/11/06 08:36:18 argiro Exp $";



class EcalWeightSetXMLTranslator {

public:
  
  EcalWeightSetXMLTranslator(){};

  int readXML  (const std::string& filename, 
		 EcalWeightSet& record);

  int writeXML (const std::string& filename, 
		const EcalWeightSet& record);
  
  void write10x10(xercesc::DOMElement* node,const EcalWeightSet& record);
  void write3x10(xercesc::DOMElement* node,const EcalWeightSet& record);


private:
  
 
};



#endif // __EcalWeightSetXMLTranslator_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; scram b"
// End:
