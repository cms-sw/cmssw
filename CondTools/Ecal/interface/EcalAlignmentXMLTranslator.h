/**
   Translates a EcalAlignmentConstant record to XML
   and vice versa   

   \author Jean Fay
   \version $Id: EcalAlignmentXMLTranslator.h,v 1.1 2010/09/14 fay Exp $
   \date 14 Sept 2010
*/

#ifndef __EcalAlignmentXMLTranslator_h_
#define __EcalAlignmentXMLTranslator_h_


#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

using namespace std;

static const char CVSId__EcalAlignmentXMLTranslator[] = 
"$Id: EcalAlignmentXMLTranslator.h,v 1.1 2010/09/14 fay Exp $";

class AlignTransform;

class EcalAlignmentXMLTranslator {

public:

  static int writeXML (const std::string& filename,
		       const EcalCondHeader& header,
		       const Alignments& record);

  static string dumpXML(const EcalCondHeader& header,
			     const Alignments& record);
};

#endif // __EcalAlignmentXMLTranslator_h_
