/****************************************************************************
 *
 * 
 * Authors: 
 *  F.Ferro ferro@ge.infn.it
 *
 ****************************************************************************/

#include "CondFormats/PPSObjects/interface/CTPPSPixelFramePosition.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>

using namespace std;

//----------------------------------------------------------------------------------------------------

std::ostream &operator<<(std::ostream &s, const CTPPSPixelFramePosition &fp) {
  return s << fp.getFEDId() << ":" << fp.getFMCId() << ":" << fp.getChannelIdx() << ":" << fp.getROC();
}

//----------------------------------------------------------------------------------------------------

void CTPPSPixelFramePosition::printXML() {
  edm::LogInfo("printXML") << "\" FEDId=\"" << getFEDId() << "\" FMCId=\"" << getFMCId() << "\" ChannelIdx=\""
                           << getChannelIdx() << "\" ROC=\"" << getROC() << "\"";
}

//----------------------------------------------------------------------------------------------------

unsigned char CTPPSPixelFramePosition::setXMLAttribute(const std::string &attribute,
                                                       const std::string &value,
                                                       unsigned char &flag) {
  unsigned int v = atoi(value.c_str());

  if (attribute == "FEDId") {
    setFEDId(v);
    flag |= 0x8;
    return 0;
  }

  if (attribute == "FMC") {
    setFMCId(v);
    flag |= 0x4;
    return 0;
  }

  if (attribute == "FEDChannel") {
    setChannelIdx(v);
    flag |= 0x2;
    return 0;
  }

  if (attribute == "ROCinChannel") {
    setROC(v);
    flag |= 0x1;
    return 0;
  }

  return 1;
}
