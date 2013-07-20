/**
   Translates a ESGain record to XML

   \version $Id: ESGainXMLTranslator.h,v 1.1 2013/01/16 12:48:07 depasse Exp $
   \date 11 Jan 2013
*/

#ifndef __ESGainXMLTranslator_h_
#define __ESGainXMLTranslator_h_


#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>


static const char CVSId__ESGainXMLTranslator[] = 
"$Id: ESGainXMLTranslator.h,v 1.1 2013/01/16 12:48:07 depasse Exp $";


class ESGain;

class ESGainXMLTranslator {

public:

  static int writeXML (const std::string& filename,
		       const EcalCondHeader& header,
		       const ESGain& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const ESGain& record);
};

#endif // __ESGainXMLTranslator_h_
