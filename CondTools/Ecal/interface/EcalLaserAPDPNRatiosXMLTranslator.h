/**
   Translates a EcalLaserAPDPNRatios record to XML
   and vice versa   
   \author Stefano Argiro'
   \version $Id: EcalLaserAPDPNRatiosXMLTranslator.h,v 1.1 2009/10/20 13:48:05 argiro Exp $
   \date 29 Jul 2008
*/

#ifndef __EcalLaserAPDPNRatiosXMLTranslator_h_
#define __EcalLaserAPDPNRatiosXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <string>

class EcalLaserAPDPNRatiosXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalLaserAPDPNRatios& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalLaserAPDPNRatios& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalLaserAPDPNRatios& record);
};

#endif  // __EcalLaserAPDPNRatiosXMLTranslator_h_
