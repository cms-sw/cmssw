// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
/*
 * $Id: MatacqRawEvent.cc,v 1.1 2009/02/25 14:44:25 pgras Exp $
 * Original author: Ph. Gras CEA/Saclay 
 */

/**
 * \file
 * Implementation of the MaacqRawEvent class
 */

#include <unistd.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <limits>

#define CMSSW

#ifdef CMSSW //compilation within CMSSW framework

#include "EventFilter/EcalRawToDigi/interface/MatacqRawEvent.h"
#include "FWCore/Utilities/interface/Exception.h"

static inline void throwExcept(const std::string& s){
  throw cms::Exception("Matacq") << s;
}

#else //compilation outside CMSSW framework (e.g. online)

#include "MatacqRawEvent.h"
#include <stdexcept>
static inline void throwExcept(const std::string& s){
  throw std::runtime_error(s.c_str());
}

#endif //CMSSW not defined

using namespace std;

//DAQ header fields:
const MatacqRawEvent::field32spec_t MatacqRawEvent::fov32             	= {0, 0x000000F0};
const MatacqRawEvent::field32spec_t MatacqRawEvent::fedId32           	= {0, 0x000FFF00};
const MatacqRawEvent::field32spec_t MatacqRawEvent::bxId32            	= {0, 0xFFF00000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::lv132             	= {1, 0x00FFFFFF};
const MatacqRawEvent::field32spec_t MatacqRawEvent::triggerType32     	= {1, 0x0F000000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::boeType32         	= {1, 0xF0000000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::dccLen32          	= {2, 0x00FFFFFF};
const MatacqRawEvent::field32spec_t MatacqRawEvent::dccErrors32       	= {2, 0xFF000000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::runNum32          	= {3, 0x00FFFFFF};
const MatacqRawEvent::field32spec_t MatacqRawEvent::h1Marker32        	= {3, 0x3F000000};

//Matacq header fields:
const MatacqRawEvent::field32spec_t MatacqRawEvent::formatVersion32   	= {4, 0x0000FFFF};
const MatacqRawEvent::field32spec_t MatacqRawEvent::freqGHz32         	= {4, 0x00FF0000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::channelCount32    	= {4, 0xFF000000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::timeStamp32       	= {5, 0xFFFFFFFF};
//  for data format version >=2:
const MatacqRawEvent::field32spec_t MatacqRawEvent::tTrigPs32         	= {6, 0xFFFFFFFF};
//  for data format version >=3:
const MatacqRawEvent::field32spec_t MatacqRawEvent::orbitId32         	= {7, 0xFFFFFFFF};
const MatacqRawEvent::field32spec_t MatacqRawEvent::vernier0_32       	= {8, 0x0000FFFF};
const MatacqRawEvent::field32spec_t MatacqRawEvent::vernier1_32       	= {8, 0xFFFF0000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::vernier2_32       	= {9, 0x0000FFFF};
const MatacqRawEvent::field32spec_t MatacqRawEvent::vernier3_32       	= {9, 0xFFFF0000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::timeStampMicroSec32 = {10,0xFFFFFFFF};
const MatacqRawEvent::field32spec_t MatacqRawEvent::trigRec32         	= {11,0xFF000000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::postTrig32        	= {11,0x0000FFFF};
const MatacqRawEvent::field32spec_t MatacqRawEvent::laserPower32        = {12,0x000000FF};
const MatacqRawEvent::field32spec_t MatacqRawEvent::attenuation_dB32    = {12,0x00000F00};
const MatacqRawEvent::field32spec_t MatacqRawEvent::emtcPhase32         = {12,0x0000F000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::emtcDelay32         = {12,0xFFFF0000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::delayA32            = {13,0x0000FFFF};
const MatacqRawEvent::field32spec_t MatacqRawEvent::dccId32             = {13,0x003F0000}; 
const MatacqRawEvent::field32spec_t MatacqRawEvent::color32             = {13,0x00600000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::trigType32          = {13,0x07000000};
const MatacqRawEvent::field32spec_t MatacqRawEvent::side32              = {13,0x08000000};

void MatacqRawEvent::setRawData(const unsigned char* pData, size_t maxSize){
  error = 0;
  unsigned char* begin = (unsigned char*) pData;
  int16le_t* begin16 = (int16le_t*) pData;
  uint32le_t* begin32 = (uint32le_t*) pData;
  int16le_t* pData16 = begin16;
  const int daqHeaderLen = 16; //in bytes
  if(maxSize < 6*4){
    error = errorLength;
    return;
  }
  pData16 += daqHeaderLen/sizeof(pData16[0]);
  //  matacqHeader = (matacqHeader_t*) pData16;
  daqHeader = begin32;
  matacqDataFormatVersion = read32(begin32, formatVersion32);
  freqGHz       	  = read32(begin32, freqGHz32);
  channelCount  	  = read32(begin32, channelCount32);
  timeStamp.tv_sec     	  = read32(begin32, timeStamp32);
  int headerLen = 24; //in bytes
  if(matacqDataFormatVersion>=2){
    tTrigPs       = read32(begin32, tTrigPs32);
    headerLen += 4;
  } else{
    tTrigPs = numeric_limits<int>::max();
  }

  if(matacqDataFormatVersion>=3){
    orbitId           = read32(begin32, orbitId32);
    vernier[0]        = read32(begin32, vernier0_32);
    vernier[1]        = read32(begin32, vernier1_32);
    vernier[2]        = read32(begin32, vernier2_32);
    vernier[3]        = read32(begin32, vernier3_32);
    timeStamp.tv_usec = read32(begin32, timeStampMicroSec32);
    trigRec           = read32(begin32, trigRec32, true);
    postTrig          = read32(begin32, postTrig32);
    delayA            = read32(begin32, delayA32, true);
    emtcDelay         = read32(begin32, emtcDelay32, true);
    emtcPhase         = read32(begin32, emtcPhase32, true);
    attenuation_dB    = read32(begin32, attenuation_dB32, true);
    laserPower        = read32(begin32, laserPower32, true);
    headerLen = 64;
  } else{
    orbitId = 0;
    vernier[0] = -1;
    vernier[1] = -1;
    vernier[2] = -1;
    vernier[3] = -1;
    trigRec = -1;
    postTrig = -1;
    delayA = -1;
    emtcDelay = -1;
    emtcPhase = -1;
    attenuation_dB = -1;
    laserPower = -1;
  }
    
  const int nCh = getChannelCount();
  channelData.resize(nCh);

  pData16 = (int16le_t*) (begin+headerLen); 

  for(int iCh=0; iCh<nCh; ++iCh){
    if((size_t)(pData16-begin16)>maxSize){
      throwExcept(string("Corrupted or truncated data"));
    }
    //channel id:
    channelData[iCh].chId = *(pData16++);
    //number of time samples for this channel:
    channelData[iCh].nSamples = *(pData16++);
    //pointer to time sample data of this channel:
    channelData[iCh].samples = pData16;
    //moves to next channel data block:
    if(channelData[iCh].nSamples<0){
      throwExcept(string("Corrupted or truncated data"));
    }
    pData16 += channelData[iCh].nSamples;
  }
  
  //data trailer chekes:
  //FED header is aligned on 64-bit=>padding to skip
  int padding = (4-(pData16-begin16))%4;
  if(padding<0) padding+=4;
  pData16 += padding;
  if((size_t)(pData16-begin16)>maxSize){
    throwExcept(string("Corrupted or truncated data"));
  }
  uint32le_t* trailer32 = (uint32le_t*)(pData16);
  fragLen = trailer32[1]&0xFFFFFF;
  
  //cout << "Event fragment length including headers: " << fragLen
  //	 << " 64-bit words\n";
  
  //FIXME: I am expecting the event length specifies in the header to
  //include the header, while it is not the case in current TB 2006 data
  const int nHeaders = 3;
  if(fragLen!=read32(begin32,dccLen32)+nHeaders
     && fragLen != read32(begin32,dccLen32)){
    //cout << "Error: fragment length is not consistent with DCC "
    //	"length\n";
    error |= errorLengthConsistency;
  }
  
  //skip trailers
  const int trailerLen = 4;
  pData16 += trailerLen;
  
  parsedLen = (pData16-begin16) / 4;

  if((pData16-begin16)!=(4*fragLen)){
    error |= errorLength;
  }

  if((size_t)(pData16-begin16)>maxSize){
    throwExcept(string("Corrupted or truncated data"));
  }

  //some checks
  if(getBoe()!=0x5){
    error |= errorWrongBoe;
  }
}

int MatacqRawEvent::read32(uint32le_t* pData, field32spec_t spec32,
			   bool ovfTrans){
  uint32_t result =  pData[spec32.offset] & spec32.mask;
  uint32_t mask = spec32.mask;
  while((mask&0x1) == 0){
      mask >>= 1;
      result >>= 1;
  }
  if(ovfTrans){
    //overflow bit (MSB) mask:
    mask = ((mask >>1) + 1);
    if(result & mask)  result = (uint32_t)-1;
  }
  return result;
}
