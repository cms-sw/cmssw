/**
   Translates a EcalTPGLinearizationConst record to XML
   \version $Id: EcalTPGLinearizationConstXMLTranslator.h,v 1.0 2012/05/01 13:48:05 Exp $
   \date 01 May 2012
*/

#ifndef __EcalTPGLinearizationConstXMLTranslator_h_
#define __EcalTPGLinearizationConstXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
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

static const char CVSId__EcalTPGLinearizationConstXMLTranslator[] = 
"$Id: EcalTPGLinearizationConstXMLTranslator.h,v 1.1 2009/10/20 13:48:05 argiro Exp $";




class EcalTPGLinearizationConstXMLTranslator {

public:

  static int writeXML (const std::string& filename,
		       const EcalCondHeader& header,
		       const EcalTPGLinearizationConst& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalTPGLinearizationConst& record);

};

#endif // __EcalTPGLinearizationConstXMLTranslator_h_
