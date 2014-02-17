/*  
 *  $Date: 2008/09/11 13:19:22 $
 *  $Revision: 1.5 $
 *  \author J. Mans -- UMD
 */
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string.h>
#include <stdint.h>

const int HcalDCCHeader::SPIGOT_COUNT = 15;

HcalDCCHeader::HcalDCCHeader() { }

unsigned int HcalDCCHeader::getTotalLengthBytes() const { 
  unsigned int totalSize=sizeof(HcalDCCHeader);
  for (int i=0; i<SPIGOT_COUNT; i++) 
    totalSize+=(spigotInfo[i]&0x3FF)*4;
  return totalSize;
}

void HcalDCCHeader::getSpigotData(int nspigot, HcalHTRData& decodeTool) const {
  edm::LogWarning("HCAL-Unpacker") << "Using unsafe getSpigotData without length controls.  Not recommended!  Replace with new getSpigotData call";
  getSpigotData(nspigot,decodeTool,10000000);
}
  

int HcalDCCHeader::getSpigotData(int nspigot, HcalHTRData& decodeTool, int validSize) const {
  const unsigned short* base=((unsigned short*)this)+sizeof(HcalDCCHeader)/sizeof(unsigned short);
  int offset=0,i,len=0;
  for (i=0; i<=nspigot; i++) {
    offset+=len;
    len=(spigotInfo[i]&0x3FF)*2;
  }
  if ((offset+len+sizeof(HcalDCCHeader)/sizeof(unsigned short))<(validSize/sizeof(unsigned short))) {
    decodeTool.adoptData(base+offset,len);
    return 0;
  } else { return -1; }
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

void HcalDCCHeader::setHeader(int sourceid, int bcn, int l1aN, int orbN) {
  commondataformat0=0x8|((sourceid&0xFFF)<<8)|((bcn&0xFFF)<<20);
  commondataformat1=0x50000000u|(l1aN&0xFFFFFF);
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
  // update the trailer...
  lenSoFar+=data.getRawLength()/2; // 32-bit words
  uint32_t* trailer=((uint32_t*)this)+sizeof(HcalDCCHeader)/sizeof(uint32_t)+lenSoFar;
  int len64=sizeof(HcalDCCHeader)/8+lenSoFar/2+1; 
  trailer[1]=0;
  trailer[0]=0xA0000000u|len64;
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
