/*  
 *  $Date: 2012/10/22 14:05:56 $
 *  $Revision: 1.1 $
 *  \author J. Mans -- UMD
 */
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDTCHeader.h"
#include <string.h>
#include <stdint.h>

const int HcalDTCHeader::SLOT_COUNT = 12;
const int HcalDTCHeader::MINIMUM_SLOT = 1;
const int HcalDTCHeader::MAXIMUM_SLOT = 12;

HcalDTCHeader::HcalDTCHeader() { }

unsigned int HcalDTCHeader::getTotalLengthBytes() const { 
  unsigned int totalSize=sizeof(HcalDTCHeader);
  for (int i=0; i<SLOT_COUNT; i++) 
    totalSize+=(slotInfo[i]&0xFFF)*sizeof(uint16_t);
  return totalSize;
}

int HcalDTCHeader::getSlotData(int nslot, HcalHTRData& decodeTool, int validSize) const {
  const unsigned short* base=((unsigned short*)this)+sizeof(HcalDTCHeader)/sizeof(unsigned short);
  int offset=0,i,len=0;
  for (i=1; i<=nslot; i++) {
    offset+=len;
    len=(slotInfo[i-1]&0xFFF);
  }
  if ((offset+len+sizeof(HcalDTCHeader)/sizeof(unsigned short))<(validSize/sizeof(unsigned short))) {
    decodeTool.adoptData(base+offset,len);
    return 0;
  } else { return -1; }
}

void HcalDTCHeader::clear() {
  commondataformat0=0;
  commondataformat1=0x50000000u;
  commondataformat2=0;
  commondataformat3=0;
  dcch0=0x1; // format version 1
  dcch1=0;
  for (int i=0; i<SLOT_COUNT; i++) 
    slotInfo[i]=0;
}

void HcalDTCHeader::setHeader(int sourceid, int bcn, int l1aN, int orbN) {
  commondataformat0=0x8|((sourceid&0xFFF)<<8)|((bcn&0xFFF)<<20);
  commondataformat1=0x50000000u|(l1aN&0xFFFFFF);
}

void HcalDTCHeader::copySlotData(unsigned int slot_id, const HcalHTRData& data, bool valid) {
  if (slot_id==0 || slot_id>(unsigned int)SLOT_COUNT) return;
  // construct the slot info
  slotInfo[slot_id-1]=(data.getRawLength())|0xc000;
  if (valid) slotInfo[slot_id-1]|=0x2000;
  // status info...
  //  if (valid) dcch0|=(1<<(slot_id+14));
  // copy
  unsigned int lenSoFar=0;
  for (unsigned int i=1; i<slot_id; i++) lenSoFar+=getSlotDataLength(i);
  unsigned short* startingPoint=((unsigned short*)this)+sizeof(HcalDTCHeader)/sizeof(unsigned short)+lenSoFar;
  memcpy(startingPoint,data.getRawData(),sizeof(unsigned short)*data.getRawLength());
  // update the trailer...
  lenSoFar+=data.getRawLength(); 
  uint32_t* trailer=((uint32_t*)this)+sizeof(HcalDTCHeader)/sizeof(uint32_t)+lenSoFar/2;
  int len64=sizeof(HcalDTCHeader)/8+lenSoFar/4+1; 
  trailer[1]=0;
  trailer[0]=0xA0000000u|len64;
}

std::ostream& operator<<(std::ostream& s, const HcalDTCHeader& head) {

  for (int i=0; i<HcalDTCHeader::SLOT_COUNT; i++) {
    s << "Slot " << i << " : " << head.getSlotDataLength(i) << " bytes, ";
    if (head.getSlotEnabled(i)) s << "E";
    if (head.getSlotPresent(i)) s << "P";
    if (head.getSlotValid(i)) s << "V";
    if (head.getSlotCRCError(i)) s << "C";
    //    s << ". Error codes: " << std::hex << int(head.getSlotErrorBits(i)) << "," << int(head.getLRBErrorBits(i)) << std::dec;
    s << std::endl;
  }
  return s;
}
