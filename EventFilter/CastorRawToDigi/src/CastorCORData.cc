//#include "Utilities/Configuration/interface/Architecture.h"
/*  
 *  $Date: 2012/09/24 11:48:43 $
 *  $Revision: 1.3 $
 *  \author A. Campbell - DESY
 */
#ifndef HTBDAQ_DATA_STANDALONE
#include "EventFilter/CastorRawToDigi/interface/CastorCORData.h"
#else
#include "CastorCORData.h"
#endif
#include <string.h>
#include <iostream>
#include <algorithm>
#include <iomanip>

using namespace std;

const int CastorCORData::CHANNELS_PER_SPIGOT         = 36;
const int CastorCORData::MAXIMUM_SAMPLES_PER_CHANNEL = 20;

CastorCORData::CastorCORData() : m_formatVersion(-2), m_rawLength(0), m_rawConst(0), m_ownData(0) { }
CastorCORData::CastorCORData(const unsigned short* data, int length) {
  adoptData(data,length);
  m_ownData=0;
}
CastorCORData::CastorCORData(const CastorCORData& hd) : m_formatVersion(hd.m_formatVersion), m_rawLength(hd.m_rawLength), m_rawConst(hd.m_rawConst), m_ownData(0) { }

CastorCORData::CastorCORData(int version_to_create) : m_formatVersion(version_to_create) {
  allocate(version_to_create);
}

void CastorCORData::allocate(int version_to_create) {
  m_formatVersion=version_to_create;
  // the needed space is for the biggest possible event...
  const int needed=0x200;
  // create a buffer big enough...
  m_ownData=new unsigned short[needed];
  m_rawLength=0;
  m_rawConst=m_ownData;
}

CastorCORData& CastorCORData::operator=(const CastorCORData& hd) {
  if (m_ownData==0) {
    m_formatVersion=hd.m_formatVersion;
    m_rawLength=hd.m_rawLength;
    m_rawConst=hd.m_rawConst;
  }
  return (*this);
}

void CastorCORData::adoptData(const unsigned short* data, int length) {
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

bool CastorCORData::check() const {
     // length checks
    //  minimum length
    if (m_rawLength<6+12) return false;
    //  matches wordcount
    if (m_rawLength!=m_rawConst[m_rawLength-3]) return false;
 
    // daq/tp length check
    int tp, daq, header, trailer, trigger;
    determineSectionLengths(tp,daq,header,trailer,trigger);
    if (trigger+daq+header+trailer>m_rawLength) return false;

  return true;
}

void CastorCORData::determineSectionLengths(int& tpWords, int& daqWords, int& headerWords, int& trailerWords, int& triggerLen) const {
 
    tpWords=m_rawConst[5]>>8; // should be 8 but could be up to 12
    if (m_rawLength>4) 
      daqWords=m_rawConst[m_rawLength-4]&0x7FF; // no zero suppression supported
	  // there are 24 16bit words per time sample
	  // these contain the data from 36 channels
	  // daqWords is number of 16 bit words of qie data 
	  // hence #qie data values id daqWords*3/2
    headerWords=8;
	triggerLen=12;   // total space reserved for trigger information
    trailerWords=12; // minimum, may be more...
}

void CastorCORData::determineStaticLengths(int& headerWords, int& trailerWords, int& triggerLen) const {
   headerWords=8;
   triggerLen=12;   // total space reserved for trigger information
   trailerWords=12; // minimum, may be more...

}


void CastorCORData::unpack(unsigned char* daq_lengths, unsigned short* daq_samples,
			 unsigned char* tp_lengths, unsigned short* tp_samples) const {

  if (daq_lengths!=0) memset(daq_lengths,0,CHANNELS_PER_SPIGOT);
  if (tp_lengths!=0) memset(tp_lengths,0,1);

  int tp_words_total = 0;
  int daq_words_total = 0;
  int headerLen = 0;
  int trailerLen = 0;
  int triggerLen = 0;
  determineSectionLengths(tp_words_total,daq_words_total,headerLen,trailerLen,triggerLen);

  int wordPtr;
  const unsigned short* tpBase=m_rawConst+headerLen;
  // process the trigger primitive words
  if (tp_lengths!=0) {
    for (wordPtr=0; wordPtr<tp_words_total; wordPtr++) {
       tp_samples[tp_lengths[0]]=tpBase[wordPtr];
       tp_lengths[0]++;
    }
  }
 
  const unsigned short* daqBase=m_rawConst+headerLen+triggerLen;
  unsigned long dat;
  // process the DAQ words
  int lastCapid=0;
  int ts,dv;
  int tsamples = daq_words_total/24;
  if (daq_lengths!=0) {
	for ( ts = 0; ts < tsamples; ts++ ) {
		for (int j=0; j<12 ; j++) {
			dat = daqBase[(ts*12+j)*2]<<16 | daqBase[(ts*12+j)*2+1];
			dv  = ( dat & 0x80000000 ) >> 31;
			daq_samples[(j*3)  *MAXIMUM_SAMPLES_PER_CHANNEL+ts]= (( dat & 0x40000000 ) >> 20 ) | (( dat & 0x3fe00000 ) >> 21 ) | ( dv << 9 );
			daq_samples[(j*3+1)*MAXIMUM_SAMPLES_PER_CHANNEL+ts]= (( dat & 0x00100000 ) >> 10 ) | (( dat & 0x000ff800 ) >> 11 ) | ( dv << 9 );
			daq_samples[(j*3+2)*MAXIMUM_SAMPLES_PER_CHANNEL+ts]= (( dat & 0x00000400 )       ) | (( dat & 0x000003fe ) >>  1 ) | ( dv << 9 );
		}
    }
   // now loop over channels - set daq_lengths with error bits
   int ichan;
   for ( ichan = 0; ichan<CHANNELS_PER_SPIGOT; ichan++) {
	   daq_lengths[ichan]=tsamples;
	   for ( ts = 0; ts < tsamples; ts++ ) {
	      int erdv =(daq_samples[ichan*MAXIMUM_SAMPLES_PER_CHANNEL+ts] & 0x600 ) >> 9;
		  int capid=(daq_samples[ichan*MAXIMUM_SAMPLES_PER_CHANNEL+ts] & 0x180 ) >> 7;
		  if ( erdv!=1 || ( ts!=0 && (capid!=((lastCapid+1)%4)))) {
			 daq_lengths[ichan]|=0x80; 
	      }
		  lastCapid=capid;
       }
   }
 }
}

void CastorCORData::pack(unsigned char* daq_lengths, unsigned short* daq_samples,
		       unsigned char* tp_lengths, unsigned short* tp_samples, bool do_capid) {
  
  int tp_words_total=0, daq_words_total=0, headerLen, trailerLen, triggerLen;
  determineStaticLengths(headerLen,trailerLen,triggerLen);

  tp_words_total=0;
  daq_words_total=0;
  int isample;

  // trigger words
  unsigned short* ptr=m_ownData+headerLen;
  if (tp_samples!=0 && tp_lengths!=0) {
      for (isample=0; isample<tp_lengths[0] && isample<12; isample++) {
	ptr[tp_words_total]=tp_samples[isample];
	tp_words_total++;
      }
  }

  // daq words
  ptr=m_ownData+headerLen+triggerLen;
  int timesamples = std::min (daq_lengths[0]&0x3f,MAXIMUM_SAMPLES_PER_CHANNEL) ;
  int ts, capid, j;
  unsigned long dat;
  unsigned short s1,s2,s3;
  bool somevalid;

  for (ts=0; ts<timesamples; ts++){
	capid = ts%4;
    for (j=0; j<12 ; j++) {
		somevalid = false;
		if ( daq_lengths[j*3] == 0 || ( daq_lengths[j*3] & 0xc0 ) ) {
			s1 = 0x400; // ER !DV
	    } else {
			s1 = daq_samples[(j*3  )*MAXIMUM_SAMPLES_PER_CHANNEL+ts];
			somevalid = true;
	    }
		if ( daq_lengths[j*3+1] == 0 || ( daq_lengths[j*3+1] & 0xc0 ) ) {
			s2 = 0x400; // ER !DV
	    } else {
			s2 = daq_samples[(j*3+1)*MAXIMUM_SAMPLES_PER_CHANNEL+ts];
			somevalid = true;
	    }
		if ( daq_lengths[j*3+2] == 0 || ( daq_lengths[j*3+2] & 0xc0 ) ) {
			s3 = 0x400; // ER !DV
	    } else {
			s3 = daq_samples[(j*3+2)*MAXIMUM_SAMPLES_PER_CHANNEL+ts];
			somevalid = true;
	    }
		//dat = 0x80000001 // msb is dv for the fibre
	                     //// sample data  is ER(1)+CAPID(2)+EXP(2)+Mantissa(5)
						 //// daq_samples has ER(1)+DV(1)+CAPID(2)+EXP(2)+Mantissa(5)
						 //// should check daq_lengths for the 3 channels here ??
		     //|  ( daq_samples[(j*3  )*MAXIMUM_SAMPLES_PER_CHANNEL+ts] & 0x1ff ) << 21
			 //|  ( daq_samples[(j*3+1)*MAXIMUM_SAMPLES_PER_CHANNEL+ts] & 0x1ff ) << 11
			 //|  ( daq_samples[(j*3+2)*MAXIMUM_SAMPLES_PER_CHANNEL+ts] & 0x1ff ) <<  1
		     //|  ( daq_samples[(j*3  )*MAXIMUM_SAMPLES_PER_CHANNEL+ts] & 0x400 ) << 20
			 //|  ( daq_samples[(j*3+1)*MAXIMUM_SAMPLES_PER_CHANNEL+ts] & 0x400 ) << 10
			 //|  ( daq_samples[(j*3+2)*MAXIMUM_SAMPLES_PER_CHANNEL+ts] & 0x400 ) ;
		dat = 0x00000001 // msb is dv for the fibre
	                     // sample data  is ER(1)+CAPID(2)+EXP(2)+Mantissa(5)
						 // daq_samples has ER(1)+DV(1)+CAPID(2)+EXP(2)+Mantissa(5)
						 // should check daq_lengths for the 3 channels here ??
		     |  ( s1 & 0x1ff ) << 21
			 |  ( s2 & 0x1ff ) << 11
			 |  ( s3 & 0x1ff ) <<  1
		     |  ( s1 & 0x400 ) << 20
			 |  ( s2 & 0x400 ) << 10
			 |  ( s3 & 0x400 ) ;
	    if ( somevalid ) dat |= 0x80000000;
		// should we set dv from daq_samples ??
		if (do_capid) dat = ( dat & 0xcff3fcff ) | capid << 28 | capid << 18 | capid << 8;
		ptr[daq_words_total++] = dat >> 16;
		ptr[daq_words_total++] = dat & 0xffff;

    } 
  }

   m_ownData[5]=(tp_words_total<<8)|0x1;
   unsigned short totalLen=headerLen+12+daq_words_total+trailerLen;

   m_rawLength=totalLen;
   m_ownData[totalLen-2]=totalLen/2; // 32-bit words
   m_ownData[totalLen-3]=totalLen;
   m_ownData[totalLen-4]=daq_words_total;
 
}

void CastorCORData::packHeaderTrailer(int L1Anumber, int bcn, int submodule, int orbitn, int pipeline, int ndd, int nps, int firmwareRev) {
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

unsigned int CastorCORData::getOrbitNumber() const { 
  return (m_rawConst[3]>>11);
}
unsigned int CastorCORData::getSubmodule() const {
  return (m_rawConst[3]&0x7FF);
}
unsigned int CastorCORData::htrSlot() const{
  const unsigned int smid = getSubmodule();
  return ((smid>>1)&0x1F);
} 
unsigned int CastorCORData::htrTopBottom() const{
  const unsigned int smid = getSubmodule();
  return (smid&0x01);
} 
unsigned int CastorCORData::readoutVMECrateId() const{
  const unsigned int smid = getSubmodule();
  return ((smid>>6)&0x1F);
} 
bool CastorCORData::isCalibrationStream() const {
  return (m_formatVersion==-1)?(false):(m_rawConst[2]&0x4000);
}
bool CastorCORData::isUnsuppressed() const {
  return (m_formatVersion<4)?(false):(m_rawConst[6]&0x8000);
}
bool CastorCORData::wasMarkAndPassZS(int fiber, int fiberchan) const {
  if (fiber<1 || fiber>8 || fiberchan<0 || fiberchan>2) return false;
  if (!isUnsuppressed() || m_formatVersion<5) return false;
  unsigned short val=(fiber<5)?(m_rawConst[m_rawLength-12]):(m_rawConst[m_rawLength-11]);
  int shift=(((fiber-1)%4)*3)+fiberchan;
  return ((val>>shift)&0x1)!=0;
} 

bool CastorCORData::isPatternRAMEvent() const {
  return (m_formatVersion==-1)?(false):(m_rawConst[2]&0x1000);
}
bool CastorCORData::isHistogramEvent() const {
  return (m_formatVersion==-1)?(m_rawConst[2]&0x2):(m_rawConst[2]&0x2000);
}
int CastorCORData::getNDD() const {
  return (m_formatVersion==-1)?(m_rawConst[m_rawLength-4]>>8):(m_rawConst[m_rawLength-4]>>11);
}
int CastorCORData::getNTP() const {
  int retval=-1;
  if (m_formatVersion==-1) retval=m_rawConst[m_rawLength-4]&0xFF;
  else if (m_formatVersion<3) retval=m_rawConst[m_rawLength-4]>>11;
  return retval;
}
int CastorCORData::getNPrecisionWords() const {
  return (m_formatVersion==-1)?(m_rawConst[m_rawLength-4]&0xFF):(m_rawConst[m_rawLength-4]&0x7FF);
}
int CastorCORData::getNPS() const {
  return (m_formatVersion==-1)?(0):((m_rawConst[5]>>4)&0xF);
}
unsigned int CastorCORData::getPipelineLength() const {
  return (m_rawConst[7]&0xFF);
}
unsigned int CastorCORData::getFirmwareRevision() const {
  return (m_rawConst[6]);
}

void CastorCORData::getHistogramFibers(int& a, int& b) const {
  a=-1;
  b=-1;
  if (m_formatVersion==-1) {
    a=((m_rawConst[2]&0x0F00)>>8);
    b=((m_rawConst[2]&0xF000)>>12);
  } else {
    a=((m_rawConst[5]&0x0F00)>>8);
    b=((m_rawConst[5]&0xF000)>>12);
  }
}

bool CastorCORData::unpackHistogram(int myfiber, int mysc, int capid, unsigned short* histogram) const {
  // check for histogram mode
  if (!isHistogramEvent()) return false;

  int fiber1, fiber2;
  getHistogramFibers(fiber1,fiber2);
  if (fiber1!=myfiber && fiber2!=myfiber) return false;

  if (m_formatVersion==-1) {
    int offset=6+mysc*4*32+capid*32;
    if (myfiber==fiber2) offset+=3*4*32; // skip to the second half...
    for (int i=0; i<32; i++)
      histogram[i]=m_rawConst[offset+i];
    return true;
  } else {
    int offset=8+mysc*4*32+capid*32;
    if (myfiber==fiber2) offset+=3*4*32; // skip to the second half...
    for (int i=0; i<32; i++)
      histogram[i]=m_rawConst[offset+i];
    return true;
  }
}

