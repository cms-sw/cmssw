/**
   Translates an EcalTimeOffsetConstant record to XML
   and vice versa   

   \author Seth Cooper, University of Minnesota
   \version $Id: EcalTimeOffsetXMLTranslator.h,v 1.1 2011/03/22 16:13:04 argiro Exp $
   \date 21 Mar 2011
*/

#ifndef __EcalTimeOffsetXMLTranslator_h_
#define __EcalTimeOffsetXMLTranslator_h_


#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>


static const char CVSId__EcalTimeOffsetXMLTranslator[] = 
"$Id: EcalTimeOffsetXMLTranslator.h,v 1.1 2011/03/22 16:13:04 argiro Exp $";


class EcalTimeOffsetConstant;

class EcalTimeOffsetXMLTranslator {

public:

  static int readXML  (const std::string& filename, 
		       EcalCondHeader& header,
		       EcalTimeOffsetConstant& record);

  static int writeXML (const std::string& filename,
		       const EcalCondHeader& header,
		       const EcalTimeOffsetConstant& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalTimeOffsetConstant& record);
};



#endif // __EcalTimeOffsetXMLTranslator_h_

