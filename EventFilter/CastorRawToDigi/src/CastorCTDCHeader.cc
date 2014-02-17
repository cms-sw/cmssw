/*  
 *  $Date: 2009/02/20 17:46:26 $
 *  $Revision: 1.1 $
 *  \author A. Campbell -- DESY
 */
#include "EventFilter/CastorRawToDigi/interface/CastorCORData.h"
#include "EventFilter/CastorRawToDigi/interface/CastorMergerData.h"
#include "EventFilter/CastorRawToDigi/interface/CastorCTDCHeader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string.h>
#include <stdint.h>

const int CastorCTDCHeader::SPIGOT_COUNT = 2; // COR spigots - does not include merger pay load

CastorCTDCHeader::CastorCTDCHeader() { }

unsigned int CastorCTDCHeader::getTotalLengthBytes() const { 
  unsigned int totalSize=sizeof(CastorCTDCHeader);
  for (int i=0; i<SPIGOT_COUNT+1; i++)    // includes merger pay load
    totalSize+=(spigotInfo[i]&0x3FF)*4;
  return totalSize; // doesn't include the trailer
}  

int CastorCTDCHeader::getSpigotData(int nspigot, CastorCORData& decodeTool, int validSize) const {
  const unsigned short* base=((unsigned short*)this)+sizeof(CastorCTDCHeader)/sizeof(unsigned short);
  int offset=0,i,len=0;
  for (i=0; i<=nspigot; i++) {
    offset+=len;
    len=(spigotInfo[i]&0x3FF)*2;
  }
  if ((offset+len+sizeof(CastorCTDCHeader)/sizeof(unsigned short))<(validSize/sizeof(unsigned short))) {
    decodeTool.adoptData(base+offset,len);
    return 0;
  } else { return -1; }
}

void CastorCTDCHeader::clear() {
  commondataformat0=0;
  commondataformat1=0x50000000u;
  commondataformat2=0;
  commondataformat3=0;
  ctdch0=0x1; // format version 1
  ctdch1=0;
  for (int i=0; i<3; i++) 
    spigotInfo[i]=0;
  spigotInfo[3]=0x12345678; // end DCC header pattern

}

void CastorCTDCHeader::setHeader(int sourceid, int bcn, int l1aN, int orbN) {
  commondataformat0=0x8|((sourceid&0xFFF)<<8)|((bcn&0xFFF)<<20);
  commondataformat1=0x50000000u|(l1aN&0xFFFFFF);
}

void CastorCTDCHeader::copySpigotData(unsigned int spigot_id, const CastorCORData& data, bool valid, unsigned char LRB_error_word) {
  if (spigot_id>=(unsigned int)SPIGOT_COUNT) return;
  // construct the spigot info
  spigotInfo[spigot_id]=(data.getRawLength()/2)|0xc000;
  if (valid) spigotInfo[spigot_id]|=0x2000;
  spigotInfo[spigot_id]|=(LRB_error_word<<16)|((data.getErrorsWord()&0xFF)<<24);
  // status info...
  if (valid) ctdch0|=(1<<(spigot_id+14));
  // copy
  unsigned int lenSoFar=0;
  for (unsigned int i=0; i<spigot_id; i++) lenSoFar+=getSpigotDataLength(i);
  unsigned short* startingPoint=((unsigned short*)this)+sizeof(CastorCTDCHeader)/sizeof(unsigned short)+lenSoFar*2;
  memcpy(startingPoint,data.getRawData(),sizeof(unsigned short)*data.getRawLength());
  // update the trailer...
  lenSoFar+=data.getRawLength()/2; // 32-bit words
  uint32_t* trailer=((uint32_t*)this)+sizeof(CastorCTDCHeader)/sizeof(uint32_t)+lenSoFar;
  int len64=sizeof(CastorCTDCHeader)/8+lenSoFar/2+1; 
  trailer[1]=0;
  trailer[0]=0xA0000000u|len64;
}

void CastorCTDCHeader::copyMergerData(const CastorMergerData& data, bool valid) {
  unsigned int spigot_id = 2;
  // construct the spigot info
  spigotInfo[spigot_id]=(data.getRawLength()/2)|0xc000; // Enabled & Present
  if (valid) spigotInfo[spigot_id]|=0x2000; // Valid
  spigotInfo[spigot_id]|=((data.getErrorsWord()&0xFF)<<24);
  // status info...
  if (valid) ctdch0|=(1<<(spigot_id+14));
  // copy
  unsigned int lenSoFar=0;
  for (unsigned int i=0; i<spigot_id; i++) lenSoFar+=getSpigotDataLength(i);
  unsigned short* startingPoint=((unsigned short*)this)+sizeof(CastorCTDCHeader)/sizeof(unsigned short)+lenSoFar*2;
  memcpy(startingPoint,data.getRawData(),sizeof(unsigned short)*data.getRawLength());
  // update the trailer...
  lenSoFar+=data.getRawLength()/2; // 32-bit words
  uint32_t* trailer=((uint32_t*)this)+sizeof(CastorCTDCHeader)/sizeof(uint32_t)+lenSoFar;
  int len64=sizeof(CastorCTDCHeader)/8+lenSoFar/2+1; 
  trailer[1]=0;
  trailer[0]=0xA0000000u|len64;
}

std::ostream& operator<<(std::ostream& s, const CastorCTDCHeader& head) {

  for (int i=0; i<CastorCTDCHeader::SPIGOT_COUNT+1; i++) {
    s << "Spigot " << i << " : " << head.getSpigotDataLength(i) << " bytes, ";
    if (head.getSpigotEnabled(i)) s << "E";
    if (head.getSpigotPresent(i)) s << "P";
    if (head.getSpigotValid(i)) s << "V";
    s << ". Error codes: " << std::hex << int(head.getSpigotErrorBits(i)) << std::dec;
    s << std::endl;
  }
  return s;
}
