/*  
 *  $Date: 2008/04/22 17:17:25 $
 *  $Revision: 1.8 $
 *  \author J. Mans -- UMN
 */
#include "EventFilter/HcalRawToDigi/interface/HcalTTPData.h"
#include <string.h>
#include <stdio.h>

HcalTTPData::HcalTTPData() {
}
HcalTTPData::HcalTTPData(const unsigned short* data, int length) : HcalHTRData(data,length) {
}
HcalTTPData::HcalTTPData(const HcalTTPData& hd) :HcalHTRData(hd) {
}

HcalTTPData::HcalTTPData(int version_to_create) : HcalHTRData(version_to_create) {
}

HcalTTPData& HcalTTPData::operator=(const HcalTTPData& hd) {
  if (m_ownData==0) {
    m_formatVersion=hd.m_formatVersion;
    m_rawLength=hd.m_rawLength;
    m_rawConst=hd.m_rawConst;
  }
  return (*this);
}

// check :: not EE, length is reasonable, length matches wordcount
//          length required for tp+daq is correct

bool HcalTTPData::check() const {
  if (m_formatVersion==-1) {
    // length checks
    //  minimum length
    if (m_rawLength<6+12) return false;
    //  matches wordcount
    if (m_rawLength!=m_rawConst[m_rawLength-3]) return false;
    // empty event check
    if (m_rawConst[2]&0x20) return false;
  } else {
    // length checks
    //  minimum length
    if (m_rawLength<8+4) return false;
    if (m_formatVersion<=3) {
      //  matches wordcount
      if (m_rawLength!=m_rawConst[m_rawLength-3]) {
	if (isHistogramEvent() && m_rawConst[m_rawLength-3]==786) {
	  // known bug!
	} else
	  return false;
      }
    } else { 
      // eventually add CRC check
    }
    // empty event check (redundant...)
    if (m_rawConst[2]&0x4) return false;
  }

  // daq/tp length check
  int daq, header, trailer;
  determineSectionLengths(daq,header,trailer);
  if (daq+header+trailer>m_rawLength) return false;

  return true;
}

static void copyBits(const unsigned short val, std::vector<bool>& ib, int from, int n=16, int offset=0) {
  for (int i=0; i<n; i++)
    ib[from+i]=((val)&(1<<(i+offset)))!=0;
}

void HcalTTPData::unpack(std::vector<InputBits>& ivs, std::vector<AlgoBits>& avs) const {
  ivs.clear();
  avs.clear();

  int dw, hw, tw;
  InputBits dummy1(TTP_INPUTS);
  AlgoBits dummy2(TTP_ALGOBITS);
  determineSectionLengths(dw,hw,tw);

  const unsigned short* workptr=m_rawConst+hw;
  for (int i=0; i<dw; i++) {   
    switch (i%6) {
    case 0: 
      ivs.push_back(dummy1);
      copyBits(workptr[i], ivs.back(),0);
      break;
    case 1: copyBits(workptr[i], ivs.back(), 16); break;
    case 2: copyBits(workptr[i], ivs.back(), 32); break;
    case 3: copyBits(workptr[i], ivs.back(), 48); break;
    case 4: 
      copyBits(workptr[i], ivs.back(), 64,8); 
      avs.push_back(dummy2);
      copyBits(workptr[i],avs.back(),0,8,8);
      break;      
    case 5: copyBits(workptr[i], avs.back(), 8); break;
    }
  }  
}

void HcalTTPData::determineSectionLengths(int& dataWords, int& headerWords, int& trailerWords) const {
  headerWords=8;
  trailerWords=4; // minimum, may be more...
  if (m_rawLength>4)
    dataWords=m_rawConst[m_rawLength-4]&0x7FF; // zero suppression supported
  else dataWords=0;
}

