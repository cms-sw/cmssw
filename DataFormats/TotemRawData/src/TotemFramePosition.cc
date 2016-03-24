/****************************************************************************
*
* This is a part of the TOTEM testbeam/monitoring software.
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/


#include "DataFormats/TotemRawData/interface/TotemFramePosition.h"

#include <iomanip>
#include <cstdlib>

using namespace std;

const string TotemFramePosition::tagSSNone ="None"; 
const string TotemFramePosition::tagSST1 ="T1";   
const string TotemFramePosition::tagSST2 ="T2";  
const string TotemFramePosition::tagSSRP ="RP"; 
const string TotemFramePosition::tagSSTrigger ="Trigger"; 
const string TotemFramePosition::tagSSTTC ="TTC"; 
const string TotemFramePosition::tagSSFEC ="FEC"; 

//----------------------------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& s, const TotemFramePosition &fp)
{
  return s << std::hex
    << fp.GetSubSystemId() << ":"
    << std::setw(2) << std::setfill('0')
    << fp.GetTOTFEDId() << ":"
    << fp.GetOptoRxId() << ":"
    << fp.GetGOHId() << ":"
    << fp.GetIdxInFiber()
    << std::dec;
}

//----------------------------------------------------------------------------------------------------

void TotemFramePosition::PrintXML()
{
  cout << dec << "SubSystemId=\"";

  const unsigned char &ss = GetSubSystemId();
  switch (ss) {
    case ssT1: cout << "T1"; break;
    case ssT2: cout << "T2"; break;
    case ssRP: cout << "RP"; break;
    case ssTrigger: cout << "Trigger"; break;
    case ssTTC: cout << "TTC"; break;
    case ssFEC: cout << "FEC"; break;
    default: cout << "None";
  }

  cout << "\" TOTFEDId=\"" << GetTOTFEDId()
    << "\" OptoRxId=\"" << GetOptoRxId()
    << "\" GOHId=\"" << GetGOHId()
    << "\" IdxInFiber=\"" << GetIdxInFiber()
    << "\"";
}

//----------------------------------------------------------------------------------------------------

unsigned char TotemFramePosition::SetXMLAttribute(const std::string &attribute, const std::string &value,
    unsigned char &flag)
{
  if (attribute.compare("DAQPosition") == 0) {
    size_t dd1 = value.find(':', 0);
    size_t dd2 = value.find(':', dd1+1);
    size_t dd3 = value.find(':', dd2+1);
    size_t dd4 = value.find(':', dd3+1);
    size_t dd5 = value.find(':', dd4+1);

    if (dd1 == string::npos || dd2 == string::npos || dd3 == string::npos || dd4 == string::npos ||
        dd5 != string::npos)
      return 3;
    
    unsigned short ss = strtol(value.substr(0, dd1).c_str(), NULL, 16);
    unsigned short ti = strtol(value.substr(dd1+1, dd2-dd1).c_str(), NULL, 16);
    unsigned short oi = strtol(value.substr(dd2+1, dd3-dd2).c_str(), NULL, 16);
    unsigned short gi = strtol(value.substr(dd3+1, dd4-dd3).c_str(), NULL, 16);
    unsigned short fi = strtol(value.substr(dd4+1).c_str(), NULL, 16);

    SetAllIDs(ss, ti, oi, gi, fi);
    flag |= 0x40;
    return 0;
  }

  if (attribute.compare("SubSystemId") == 0) {
    if (value.compare(tagSSNone) == 0) SetSubSystemId(ssNone);
    else if (value.compare(tagSST1) == 0) SetSubSystemId(ssT1);
      else if (value.compare(tagSST2) == 0) SetSubSystemId(ssT2);
        else if (value.compare(tagSSRP) == 0) SetSubSystemId(ssRP);
          else if (value.compare(tagSSTrigger) == 0) SetSubSystemId(ssTrigger);
            else if (value.compare(tagSSTTC) == 0) SetSubSystemId(ssTTC);
              else if (value.compare(tagSSFEC) == 0) SetSubSystemId(ssFEC);
                else return 2;

    flag |= 0x10;
    return 0;
  }

  unsigned int v = atoi(value.c_str());

  if (attribute.compare("TOTFEDId") == 0) {
    SetTOTFEDId(v);
    flag |= 0x8;
    return 0;
  }

  if (attribute.compare("OptoRxId") == 0) {
    SetOptoRxId(v);
    flag |= 0x4;
    return 0;
  }

  if (attribute.compare("GOHId") == 0) {
    SetGOHId(v);
    flag |= 0x2;
    return 0;
  }

  if (attribute.compare("IdxInFiber") == 0) {
    SetIdxInFiber(v);
    flag |= 0x1;
    return 0;
  }

  return 1;
}
