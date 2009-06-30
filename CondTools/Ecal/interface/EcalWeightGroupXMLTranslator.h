/**
   Translates a EcalWeightGroup record to XML
   and vice versa   

   \author Stefano ARGIRO
   \version $Id: EcalWeightGroupXMLTranslator.h,v 1.1 2008/11/14 15:46:05 argiro Exp $
   \date 20 Jun 2008
*/

#ifndef __EcalWeightGroupXMLTranslator_h_
#define __EcalWeightGroupXMLTranslator_h_

#include "CondTools/Ecal/interface/XMLTags.h"
#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include <xercesc/dom/DOMNode.hpp>
#include <string>


static const char CVSId__EcalWeightGroupXMLTranslator[] = 
"$Id: EcalWeightGroupXMLTranslator.h,v 1.1 2008/11/14 15:46:05 argiro Exp $";



class EcalWeightGroupXMLTranslator {

public:
  

  static int readXML  (const std::string& filename, 
		       EcalCondHeader& header,
		       EcalWeightXtalGroups& record);

  static int writeXML (const std::string& filename,
		       const EcalCondHeader& header,
		       const EcalWeightXtalGroups& record);

  static std::string dumpXML (const EcalCondHeader& header,
			      const EcalWeightXtalGroups& record);

};



#endif // __EcalWeightGroupXMLTranslator_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; scram b"
// End:
