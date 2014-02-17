/*  
 *  $Date: 2009/12/28 23:16:05 $
 *  $Revision: 1.2 $
 *  \author A. Campbell - DESY
 */
#ifndef HTBDAQ_DATA_STANDALONE
#include "EventFilter/CastorRawToDigi/interface/CastorMergerData.h"
#else
#include "CastorMergerData.h"
#endif
#include <string.h>
#include <stdio.h>
#include <algorithm>

CastorMergerData::CastorMergerData() : m_formatVersion(-2), m_rawLength(0), m_rawConst(0), m_ownData(0) { }
CastorMergerData::CastorMergerData(const unsigned short* data, int length) {
  adoptData(data,length);
  m_ownData=0;
}
CastorMergerData::CastorMergerData(const CastorMergerData& hd) : m_formatVersion(hd.m_formatVersion), m_rawLength(hd.m_rawLength), m_rawConst(hd.m_rawConst), m_ownData(0) { }

CastorMergerData::CastorMergerData(int version_to_create) : m_formatVersion(version_to_create) {
  allocate(version_to_create);
}

void CastorMergerData::allocate(int version_to_create) {
  m_formatVersion=version_to_create;
  // the needed space is for the biggest possible event...
  const int needed=0x200;
  // create a buffer big enough...
  m_ownData=new unsigned short[needed];
  m_rawLength=0;
  m_rawConst=m_ownData;
}

CastorMergerData& CastorMergerData::operator=(const CastorMergerData& hd) {
  if (m_ownData==0) {
    m_formatVersion=hd.m_formatVersion;
    m_rawLength=hd.m_rawLength;
    m_rawConst=hd.m_rawConst;
  }
  return (*this);
}

void CastorMergerData::adoptData(const unsigned short* data, int length) {
  m_rawLength=length;
  m_rawConst=data;
  if (m_rawLength<5) {
    m_formatVersion=-2; // invalid!
  } else {
    m_formatVersion=(m_rawConst[4]>>12)&0xF;
  }
}

// check :: not EE, length is reasonable, length matches wordcount
//          length required for tp+daq is correct

bool CastorMergerData::check() const {
     // length checks
    //  minimum length
    if (m_rawLength<6+12) return false;
    //  matches wordcount
    if (m_rawLength!=m_rawConst[m_rawLength-3]) return false;
 
  return true;
}


void CastorMergerData::unpack(
			 unsigned char* tp_lengths, unsigned short* tp_samples) const {

  if (tp_lengths!=0) memset(tp_lengths,0,1);

  int tp_words_total,headerLen,trailerLen;
  determineSectionLengths(tp_words_total,headerLen,trailerLen);

  int wordPtr;
  const unsigned short* tpBase=m_rawConst+headerLen;
  // process the trigger primitive words
  if (tp_lengths!=0) {
    for (wordPtr=0; wordPtr<tp_words_total; wordPtr++) {
       tp_samples[tp_lengths[0]]=tpBase[wordPtr];
       tp_lengths[0]++;
    }
  }
 
}
void CastorMergerData::determineSectionLengths(int& tpWords, int& headerWords, int& trailerWords) const {
 
    tpWords=m_rawConst[5]>>8; // should be 8 but could be up to 12
    headerWords=8;
    trailerWords=0; // minimum, may be more...
}

void CastorMergerData::determineStaticLengths(int& headerWords, int& trailerWords) const {
   headerWords=8;
   trailerWords=0; // minimum, may be more...

}
void CastorMergerData::pack(
		       unsigned char* tp_lengths, unsigned short* tp_samples) {
  
  int tp_words_total=0, headerLen, trailerLen;
  determineStaticLengths(headerLen,trailerLen);

  tp_words_total=0;
  int isample;

  // trigger words
  unsigned short* ptr=m_ownData+headerLen;
  if (tp_samples!=0 && tp_lengths!=0) {
      for (isample=0; isample<tp_lengths[0] && isample<12; isample++) {
	ptr[tp_words_total]=tp_samples[isample];
	tp_words_total++;
      }
  }

   m_ownData[5]=(tp_words_total<<8)|0x1;
   unsigned short totalLen=headerLen+tp_words_total+trailerLen;

   m_rawLength=totalLen;
   m_ownData[totalLen-2]=totalLen/2; // 32-bit words
   m_ownData[totalLen-3]=totalLen;
   m_ownData[totalLen-4]=tp_words_total;
 
}

void CastorMergerData::packHeaderTrailer(int L1Anumber, int bcn, int submodule, int orbitn, int pipeline, int ndd, int nps, int firmwareRev) {
  m_ownData[0]=L1Anumber&0xFF;
  m_ownData[1]=(L1Anumber&0xFFFF00)>>8;

  m_ownData[2]=0x8000; // Version is valid, no error bits - status bits need definition
  m_ownData[3]=((orbitn&0x1F)<<11)|(submodule&0x7FF);
  m_ownData[4]=((m_formatVersion&0xF)<<12)|(bcn&0xFFF);
  m_ownData[5]|=((nps&0xF)<<4)|0x1;
  m_ownData[6]=((firmwareRev&0x70000)>>3)|(firmwareRev&0x1FFF);
  m_ownData[7]=(pipeline&0xFF) | ((ndd&0x1F)<<8);
  m_ownData[m_rawLength-4]&=0x7FF;
  m_ownData[m_rawLength-4]|=(ndd&0x1F)<<11;
  
  m_ownData[m_rawLength-2]=m_rawLength/2; // 32-bit words
  m_ownData[m_rawLength-1]=(L1Anumber&0xFF)<<8;
}

unsigned int CastorMergerData::getOrbitNumber() const { 
  return (m_rawConst[3]>>11);
}

unsigned int CastorMergerData::getFirmwareRevision() const {
  return (m_rawConst[6]);
}




