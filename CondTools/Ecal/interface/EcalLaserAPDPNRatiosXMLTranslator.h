/**
   Translates a EcalLaserAPDPNRatios record to XML
   and vice versa   
   \author Stefano Argiro'
   \version $Id: EcalLaserAPDPNRatiosXMLTranslator.h,v 1.2 2012/05/12 15:33:28 fay Exp $
   \date 29 Jul 2008
*/

#ifndef __EcalLaserAPDPNRatiosXMLTranslator_h_
#define __EcalLaserAPDPNRatiosXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondTools/Ecal/interface/XMLTags.h"
#include "CondTools/Ecal/interface/XercesString.h"
#include <string>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

static const char CVSId__EcalLaserAPDPNRatiosXMLTranslator[] = 
"$Id: EcalLaserAPDPNRatiosXMLTranslator.h,v 1.2 2012/05/12 15:33:28 fay Exp $";




class EcalLaserAPDPNRatiosXMLTranslator {

public:

  static int readXML  (const std::string& filename,
		       EcalCondHeader& header,
		       EcalLaserAPDPNRatios& record);

  static int writeXML (const std::string& filename,
		       const EcalCondHeader& header,
		       const EcalLaserAPDPNRatios& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalLaserAPDPNRatios& record);

  //  void WriteNodeWithTime(xercesc::DOMNode* node, const std::string& value, long int& time);
  //  static void WriteNodeWithTime(xercesc::DOMNode* node);
};



#endif // __EcalLaserAPDPNRatiosXMLTranslator_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; scram b"
// End:
