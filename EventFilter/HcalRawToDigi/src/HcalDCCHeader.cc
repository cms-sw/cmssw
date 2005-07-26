/*  
 *  $Date: 2005/06/06 19:29:37 $
 *  $Revision: 1.1 $
 *  \author J. Mans -- UMD
 */
#ifndef HTBDAQ_DATA_STANDALONE
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#else
#include "HcalHTRData.h"
#include "HcalDCCHeader.h"
#endif
#include <string.h>

const int HcalDCCHeader::SPIGOT_COUNT = 15;

HcalDCCHeader::HcalDCCHeader() { }

unsigned int HcalDCCHeader::getTotalLengthBytes() const { 
  unsigned int totalSize=sizeof(HcalDCCHeader);
  for (int i=0; i<SPIGOT_COUNT; i++) 
    totalSize+=(spigotInfo[i]&0x3FF)*4;
  return totalSize;
}

void HcalDCCHeader::getSpigotData(int nspigot, HcalHTRData& decodeTool) const {
  const unsigned short* base=((unsigned short*)this)+sizeof(HcalDCCHeader)/sizeof(unsigned short);
  int offset=0,i,len=0;
  for (i=0; i<=nspigot; i++) {
    offset+=len;
    len=(spigotInfo[i]&0x3FF)*2;
  }
  decodeTool.adoptData(base+offset,len);
}

void HcalDCCHeader::clear() {
  commondataformat0=0;
  commondataformat1=0x50000000u;
  commondataformat2=0;
  commondataformat3=0;
  dcch0=0x1; // format version 1
  dcch1=0;
  for (int i=0; i<18; i++) 
    spigotInfo[i]=0;
}

void HcalDCCHeader::copySpigotData(unsigned int spigot_id, const HcalHTRData& data, bool valid, unsigned char LRB_error_word) {
  if (spigot_id>=(unsigned int)SPIGOT_COUNT) return;
  // construct the spigot info
  spigotInfo[spigot_id]=(data.getRawLength()/2)|0xc000;
  if (valid) spigotInfo[spigot_id]|=0x2000;
  spigotInfo[spigot_id]|=(LRB_error_word<<16)|((data.getErrorsWord()&0xFF)<<24);
  // status info...
  if (valid) dcch0|=(1<<(spigot_id+14));
  // copy
  unsigned int lenSoFar=0;
  for (unsigned int i=0; i<spigot_id; i++) lenSoFar+=getSpigotDataLength(i);
  unsigned short* startingPoint=((unsigned short*)this)+sizeof(HcalDCCHeader)/sizeof(unsigned short)+lenSoFar*2;
  memcpy(startingPoint,data.getRawData(),sizeof(unsigned short)*data.getRawLength());
}

std::ostream& operator<<(std::ostream& s, const HcalDCCHeader& head) {

  for (int i=0; i<HcalDCCHeader::SPIGOT_COUNT; i++) {
    s << "Spigot " << i << " : " << head.getSpigotDataLength(i) << " bytes, ";
    if (head.getSpigotEnabled(i)) s << "E";
    if (head.getSpigotPresent(i)) s << "P";
    if (head.getSpigotValid(i)) s << "V";
    s << ". Error codes: " << std::hex << int(head.getSpigotErrorBits(i)) << "," << int(head.getLRBErrorBits(i)) << std::dec;
    s << std::endl;
  }
  return s;
}
