/**
   Translates a EcalChannelStatus record to XML
   and vice versa   
   \author Francesco RUBBO
   \version $Id: EcalChannelStatusXMLTranslator.h,v 1.2 2009/06/30 14:40:11 argiro Exp $
   \date 29 Jul 2008
*/

#ifndef __EcalChannelStatusXMLTranslator_h_
#define __EcalChannelStatusXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondTools/Ecal/interface/XMLTags.h"
#include "CondTools/Ecal/interface/XercesString.h"
#include <string>
#include <xercesc/dom/DOMNode.hpp>

static const char CVSId__EcalChannelStatusXMLTranslator[] = 
"$Id: EcalChannelStatusXMLTranslator.h,v 1.2 2009/06/30 14:40:11 argiro Exp $";




class EcalChannelStatusXMLTranslator {

public:

  static int readXML  (const std::string& filename,
		       EcalCondHeader& header,
		       EcalChannelStatus& record);

  static int writeXML (const std::string& filename,
		       const EcalCondHeader& header,
		       const EcalChannelStatus& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalChannelStatus& record);  

};



#endif // __EcalChannelStatusXMLTranslator_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; scram b"
// End:
