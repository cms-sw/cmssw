/**
   Translates a EcalIntercalibConstants record to XML
   and vice versa   

   \author Stefano ARGIRO
   \version $Id: EcalIntercalibConstantsXMLTranslator.h,v 1.1 2008/11/06 08:36:18 argiro Exp $
   \date 20 Jun 2008
*/

#ifndef __EcalIntercalibConstantsXMLTranslator_h_
#define __EcalIntercalibConstantsXMLTranslator_h_


#include "CondTools/Ecal/interface/XercesString.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include <xercesc/dom/DOMNode.hpp>
#include <string>


static const char CVSId__EcalIntercalibConstantsXMLTranslator[] = 
"$Id: EcalIntercalibConstantsXMLTranslator.h,v 1.1 2008/11/06 08:36:18 argiro Exp $";



class EcalIntercalibConstantsXMLTranslator {

public:
  
  EcalIntercalibConstantsXMLTranslator(){};

  static int readXML  (const std::string& filename,
		       EcalCondHeader& header,
		       EcalIntercalibConstants& record, 
		       EcalIntercalibErrors& error);

  static int writeXML (const std::string& filename, 
		       const EcalCondHeader& header,
		       const EcalIntercalibConstants& record, 
		       const EcalIntercalibErrors&    error );
  


};



#endif // __EcalIntercalibConstantsXMLTranslator_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; scram b"
// End:
