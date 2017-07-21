#include "EventFilter/CTPPSRawToDigi/interface/CTPPSPixelDataFormatter.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelROC.h" //KS

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <sstream>
#include <iostream>

using namespace std;
using namespace edm;

namespace {
  constexpr int m_LINK_bits = 6;
  constexpr int m_ROC_bits  = 5;
  constexpr int m_DCOL_bits = 5;
  constexpr int m_PXID_bits = 8;
  constexpr int m_ADC_bits  = 8;
}

CTPPSPixelDataFormatter::CTPPSPixelDataFormatter(std::map<CTPPSPixelFramePosition, CTPPSPixelROCInfo> const &mapping)  :  theWordCounter(0), mapping_(mapping)
{
  int s32 = sizeof(Word32);
  int s64 = sizeof(Word64);
  int s8  = sizeof(char);
  if ( s8 != 1 || s32 != 4*s8 || s64 != 2*s32) {
    LogError("UnexpectedSizes")
      <<" unexpected sizes: "
      <<"  size of char is: " << s8
      <<", size of Word32 is: " << s32
      <<", size of Word64 is: " << s64
      <<", send exception" ;
  }


  m_ADC_shift  = 0;
  m_PXID_shift = m_ADC_shift + m_ADC_bits;
  m_DCOL_shift = m_PXID_shift + m_PXID_bits;
  m_ROC_shift  = m_DCOL_shift + m_DCOL_bits;


  m_LINK_shift = m_ROC_shift + m_ROC_bits;
  m_LINK_mask = ~(~CTPPSPixelDataFormatter::Word32(0) << m_LINK_bits);
  m_ROC_mask  = ~(~CTPPSPixelDataFormatter::Word32(0) << m_ROC_bits);    

  maxROCIndex=3; 

  m_DCOL_mask = ~(~CTPPSPixelDataFormatter::Word32(0) << m_DCOL_bits);
  m_PXID_mask = ~(~CTPPSPixelDataFormatter::Word32(0) << m_PXID_bits);
  m_ADC_mask  = ~(~CTPPSPixelDataFormatter::Word32(0) << m_ADC_bits);

}

void CTPPSPixelDataFormatter::interpretRawData(  bool& errorsInEvent, int fedId, const FEDRawData& rawData, Collection & digis)
{

  int nWords = rawData.size()/sizeof(Word64);
  if (nWords==0) return;

/// check CRC bit
  const Word64* trailer = reinterpret_cast<const Word64* >(rawData.data())+(nWords-1);  
  if(!errorcheck.checkCRC(errorsInEvent, fedId, trailer)) return;

/// check headers
  const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); header--;
  bool moreHeaders = true;
  while (moreHeaders) {
    header++;
    LogTrace("")<<"HEADER:  " <<  print(*header);
    bool headerStatus = errorcheck.checkHeader(errorsInEvent, fedId, header);
    moreHeaders = headerStatus;
  }

/// check trailers
  bool moreTrailers = true;
  trailer++;
  while (moreTrailers) {
    trailer--;
    LogTrace("")<<"TRAILER: " <<  print(*trailer);
    bool trailerStatus = errorcheck.checkTrailer(errorsInEvent, fedId, nWords, trailer);
    moreTrailers = trailerStatus;
  }

/// data words
  theWordCounter += 2*(nWords-2);
  LogTrace("")<<"data words: "<< (trailer-header-1);

  int link = -1;
  int roc  = -1;

  bool skipROC=false;

  edm::DetSet<CTPPSPixelDigi> * detDigis=nullptr;

  const  Word32 * bw =(const  Word32 *)(header+1);
  const  Word32 * ew =(const  Word32 *)(trailer);
  if ( *(ew-1) == 0 ) { ew--;  theWordCounter--;}
  for (auto word = bw; word < ew; ++word) {
    LogTrace("")<<"DATA: " <<  print(*word);

    auto ww = *word;
    if unlikely(ww==0) { theWordCounter--; continue;}
    int nlink = (ww >> m_LINK_shift) & m_LINK_mask; 
    int nroc  = (ww >> m_ROC_shift) & m_ROC_mask;

    int FMC = 0;

    int convroc = nroc-1;
    CTPPSPixelFramePosition fPos(fedId, FMC, nlink, convroc);
    std::map<CTPPSPixelFramePosition, CTPPSPixelROCInfo>::const_iterator mit;
    mit = mapping_.find(fPos);

    if (mit == mapping_.end()){      
      if((nroc-1)>=maxROCIndex){
	errorcheck.checkROC(errorsInEvent, fedId,  ww); // check kind of error
      }else{
	edm::LogError("")<< " CTPPS Pixel DAQ map error " ;
      }
      continue; //skip word
    }

    CTPPSPixelROCInfo rocInfo = (*mit).second;

    CTPPSPixelROC rocp(rocInfo.iD, rocInfo.roc, convroc);

    if ( (nlink!=link) | (nroc!=roc) ) {  // new roc
      link = nlink; roc=nroc;

      skipROC = likely((roc-1)<maxROCIndex) ? false : !errorcheck.checkROC(errorsInEvent, fedId,  ww); 
      if (skipROC) continue;

      auto rawId = rocp.rawId();

      detDigis = &digis.find_or_insert(rawId);
      if ( (*detDigis).empty() ) (*detDigis).data.reserve(32); // avoid the first relocations

    }
  
    int adc  = (ww >> m_ADC_shift) & m_ADC_mask;
 

    int dcol = (ww >> m_DCOL_shift) & m_DCOL_mask;
    int pxid = (ww >> m_PXID_shift) & m_PXID_mask;

    std::pair<int,int> rocPixel;
    std::pair<int,int> modPixel;

    rocPixel = std::make_pair(dcol,pxid);

    modPixel = rocp.toGlobalfromDcol(rocPixel);

    CTPPSPixelDigi testdigi(modPixel.first, modPixel.second, adc);

    if(detDigis)
    (*detDigis).data.emplace_back( modPixel.first, modPixel.second, adc); 
 
  }

}


std::string CTPPSPixelDataFormatter::print(const  Word64 & word) const
{
  ostringstream str;
  str  <<"word64:  " << reinterpret_cast<const bitset<64>&> (word);
  return str.str();
}
