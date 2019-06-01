/**
   Translates a EcalADCToGeVConstant record to XML
   and vice versa   

   \author Stefano ARGIRO
   \version $Id: EcalADCToGeVXMLTranslator.h,v 1.1 2008/11/14 15:46:05 argiro Exp $
   \date 20 Jun 2008
*/

#ifndef __EcalADCToGeVXMLTranslator_h_
#define __EcalADCToGeVXMLTranslator_h_

#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

class EcalADCToGeVConstant;

class EcalADCToGeVXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalADCToGeVConstant& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalADCToGeVConstant& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalADCToGeVConstant& record);
};

#endif  // __EcalADCToGeVXMLTranslator_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd ..; scram b"
// End:
