/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "CondFormats/CTPPSReadoutObjects/interface/TotemFramePosition.h"

#include <iomanip>
#include <cstdlib>

using namespace std;

//----------------------------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& s, const TotemFramePosition &fp)
{
  return s
    << fp.getFEDId() << ":"
    << fp.getGOHId() << ":"
    << fp.getIdxInFiber();
}

//----------------------------------------------------------------------------------------------------

void TotemFramePosition::printXML()
{
  cout << "\" FEDId=\"" << getFEDId()
    << "\" GOHId=\"" << getGOHId()
    << "\" IdxInFiber=\"" << getIdxInFiber()
    << "\"";
}

//----------------------------------------------------------------------------------------------------

unsigned char TotemFramePosition::setXMLAttribute(const std::string &attribute, const std::string &value,
    unsigned char &flag)
{
  unsigned int v = atoi(value.c_str());

  if (attribute == "FEDId")
  {
    setFEDId(v);
    flag |= 0x1C; // SubSystem + TOTFED + OptoRx
    return 0;
  }

  if (attribute == "SubSystemId")
  {
    setSubSystemId(v);
    flag |= 0x10;
    return 0;
  }

  if (attribute == "TOTFEDId")
  {
    setTOTFEDId(v);
    flag |= 0x8;
    return 0;
  }

  if (attribute == "OptoRxId")
  {
    setOptoRxId(v);
    flag |= 0x4;
    return 0;
  }

  if (attribute == "GOHId")
  {
    setGOHId(v);
    flag |= 0x2;
    return 0;
  }

  if (attribute == "IdxInFiber")
  {
    setIdxInFiber(v);
    flag |= 0x1;
    return 0;
  }

  return 1;
}
