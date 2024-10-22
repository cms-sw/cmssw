/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "CondFormats/PPSObjects/interface/TotemT2FramePosition.h"

#include <iomanip>
#include <cstdlib>

using namespace std;

//----------------------------------------------------------------------------------------------------

std::ostream &operator<<(std::ostream &s, const TotemT2FramePosition &fp) {
  return s << fp.getFEDId() << ":" << fp.getGOHId() << ":" << fp.getIdxInFiber() << ":" << fp.getPayload();
}

//----------------------------------------------------------------------------------------------------

void TotemT2FramePosition::printXML() {
  cout << "\" FEDId=\"" << getFEDId() << "\" GOHId=\"" << getGOHId() << "\" IdxInFiber=\"" << getIdxInFiber()
       << "\" pay=\"" << getPayload() << "\"";
}

//----------------------------------------------------------------------------------------------------

unsigned char TotemT2FramePosition::setXMLAttribute(const std::string &attribute,
                                                    const std::string &value,
                                                    unsigned char &flag) {
  unsigned int v = atoi(value.c_str());

  if (attribute == "FEDId") {
    setFEDId(v);
    flag |= 0x1C;  // SubSystem + TOTFED + OptoRx
    return 0;
  }

  if (attribute == "pay") {
    setPayload(v);
    flag |= 0x20;  //T2 payload
    return 0;
  }

  if (attribute == "GOHId") {
    setGOHId(v);
    flag |= 0x2;
    return 0;
  }

  if (attribute == "IdxInFiber") {
    setIdxInFiber(v);
    flag |= 0x1;
    return 0;
  }

  return 1;
}
