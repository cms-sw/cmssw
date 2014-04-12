/**
   Translates a EcalTimeBiasCorrections record to XML and vice versa
*/

#ifndef __EcalTimeBiasCorrections_h_
#define __EcalTimeBiasCorrections_h_


#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>


static const char CVSId__EcalTimeBiasCorrectionsXMLTranslator[] = 
"$Id: EcalTimeBiasCorrections.h,v 1.1 2013/11/3 fay Exp $";


class EcalTimeBiasCorrections;

class EcalTimeBiasCorrectionsXMLTranslator {

public:

  static int readXML  (const std::string& filename, 
		       EcalCondHeader& header,
		       EcalTimeBiasCorrections& record);

  static int writeXML (const std::string& filename,
		       const EcalCondHeader& header,
		       const EcalTimeBiasCorrections& record);

  static std::string dumpXML(const EcalCondHeader& header,
			     const EcalTimeBiasCorrections& record);
};



#endif // __EcalTimeBiasCorrectionsr_h_
