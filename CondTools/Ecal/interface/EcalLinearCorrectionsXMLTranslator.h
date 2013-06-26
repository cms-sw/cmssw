/**
   Translates a EcalLinearCorrections record to XML
   and vice versa   

   \author Stefano ARGIRO
   \version $Id: EcalLinearCorrectionsXMLTranslator.h,v 1.2 2013/03/07 15:10:37 fra Exp $
   \date 20 Jun 2008
*/
#ifndef __EcalLinearCorrectionsXMLTranslator_h_
#define __EcalLinearCorrectionsXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
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

static const char CVSId__EcalLinearCorrectionsXMLTranslator[] = 
"$Id: EcalLinearCorrectionsXMLTranslator.h,v 1.2 2013/03/07 15:10:37 fra Exp $";

class EcalLinearCorrectionsXMLTranslator {

public:

  static int readXML  (const std::string& filename,
		       EcalCondHeader& header,
		       EcalLinearCorrections& record);

  static int writeXML (const std::string& filename,
		       const EcalCondHeader& header,
		       const EcalLinearCorrections& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalLinearCorrections& record);

  //  void WriteNodeWithTime(xercesc::DOMNode* node, const std::string& value, long int& time);
  //  static void WriteNodeWithTime(xercesc::DOMNode* node);
};


#endif // __EcalLinearCorrectionsXMLTranslator_h_
