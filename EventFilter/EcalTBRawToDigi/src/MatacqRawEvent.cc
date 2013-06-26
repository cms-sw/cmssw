// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
/*
 * $Id: MatacqRawEvent.cc,v 1.5 2010/08/06 20:24:29 wmtan Exp $
 * Original author: Ph. Gras CEA/Saclay 
 */

/**
 * \file
 * Implementation of the MatacqTBRawEvent class
 */

#include <unistd.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <limits>
#include <stdexcept>
#include "EventFilter/EcalTBRawToDigi/src/MatacqRawEvent.h"


const MatacqTBRawEvent::field32spec_t MatacqTBRawEvent::fov32             = {0, 0x000000F0};
const MatacqTBRawEvent::field32spec_t MatacqTBRawEvent::fedId32           = {0, 0x000FFF00};
const MatacqTBRawEvent::field32spec_t MatacqTBRawEvent::bxId32            = {0, 0xFFF00000};
const MatacqTBRawEvent::field32spec_t MatacqTBRawEvent::lv132             = {1, 0x00FFFFFF};
const MatacqTBRawEvent::field32spec_t MatacqTBRawEvent::triggerType32     = {1, 0x0F000000};
const MatacqTBRawEvent::field32spec_t MatacqTBRawEvent::boeType32         = {1, 0xF0000000};
const MatacqTBRawEvent::field32spec_t MatacqTBRawEvent::dccLen32          = {2, 0x00FFFFFF};
const MatacqTBRawEvent::field32spec_t MatacqTBRawEvent::dccErrors32       = {2, 0xFF000000};
const MatacqTBRawEvent::field32spec_t MatacqTBRawEvent::runNum32          = {3, 0x00FFFFFF};
const MatacqTBRawEvent::field32spec_t MatacqTBRawEvent::h1Marker32        = {3, 0xF0000000};

void MatacqTBRawEvent::setRawData(const unsigned char* pData, size_t maxSize){
  error = 0;
  int16le_t* begin16 = (int16le_t*) pData;
  int16le_t* pData16 = begin16;
  daqHeader = (uint32le_t*) pData16;
  const int daqHeaderLen = 16; //in bytes 
  pData16 += daqHeaderLen/sizeof(pData16[0]);
  matacqHeader = (matacqHeader_t*) pData16;
  pData16 += sizeof(matacqHeader_t)/sizeof(pData16[0]);
  if(getMatacqDataFormatVersion()>=2){//trigger position present
    tTrigPs = *((int32_t*) pData16);
    pData16 += 2;
  } else{
    tTrigPs = std::numeric_limits<int>::max();    
  }
  const int nCh = getChannelCount();
  channelData.resize(nCh);
  for(int iCh=0; iCh<nCh; ++iCh){
    //channel id:
    channelData[iCh].chId = *(pData16++);
    //number of time samples for this channel:
    channelData[iCh].nSamples = *(pData16++);
    //pointer to time sample data of this channel:
    channelData[iCh].samples = pData16;
    //moves to next channel data block:
    pData16 += channelData[iCh].nSamples;
  }
  
  //data trailer chekes:
  //FED header is aligned on 64-bit=>padding to skip
  int padding = (4-(pData16-begin16))%4;
  if(padding<0) padding+=4;
  pData16 += padding;
  uint32le_t* trailer32 = (uint32le_t*)(pData16);
  fragLen = trailer32[1]&0xFFFFFF;
  
  //std::cout << "Event fragment length including headers: " << fragLen
  //	 << " 64-bit words\n";
  
  //FIXME: I am expecting the event length specifies in the header to
  //include the header, while it is not the case in current TB 2006 data
  const int nHeaders = 3;
  if(fragLen!=read32(daqHeader,dccLen32)+nHeaders
     && fragLen != read32(daqHeader,dccLen32)){
    //std::cout << "Error: fragment length is not consistent with DCC "
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
    throw std::runtime_error(std::string("Corrupted or truncated data"));
  }

  //some checks
  if(getBoe()!=0x5){
    error |= errorWrongBoe;
  }
}

int MatacqTBRawEvent::read32(uint32le_t* pData, field32spec_t spec32) const{
  int result =  pData[spec32.offset] & spec32.mask;
  int mask = spec32.mask;
  while((mask&0x1) == 0){
      mask >>= 1;
      result >>= 1;
    }
  return result;
}
