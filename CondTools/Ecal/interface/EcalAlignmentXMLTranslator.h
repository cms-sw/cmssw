/**
   Translates a EcalAlignmentConstant record to XML
   and vice versa   

   \author Jean Fay
   \version $Id: EcalAlignmentXMLTranslator.h,v 1.2 2011/03/24 08:39:02 gowdy Exp $
   \date 14 Sept 2010
*/

#ifndef __EcalAlignmentXMLTranslator_h_
#define __EcalAlignmentXMLTranslator_h_


#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

static const char CVSId__EcalAlignmentXMLTranslator[] = 
"$Id: EcalAlignmentXMLTranslator.h,v 1.2 2011/03/24 08:39:02 gowdy Exp $";

class AlignTransform;

class EcalAlignmentXMLTranslator {

public:

  static int writeXML (const std::string& filename,
		       const EcalCondHeader& header,
		       const Alignments& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const Alignments& record);
};

#endif // __EcalAlignmentXMLTranslator_h_
