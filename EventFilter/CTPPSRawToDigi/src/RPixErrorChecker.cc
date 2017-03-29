#include "EventFilter/CTPPSRawToDigi/interface/RPixErrorChecker.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <sstream>
#include <iostream>

using namespace std;
using namespace edm;

namespace {
  constexpr int CRC_bits = 1;
//  constexpr int LINK_bits = 6;
  constexpr int ROC_bits  = 5;
  constexpr int DCOL_bits = 5;
  constexpr int PXID_bits = 8;
  constexpr int ADC_bits  = 8;
  constexpr int OMIT_ERR_bits = 1;
  
  constexpr int CRC_shift = 2;
  constexpr int ADC_shift  = 0;
  constexpr int PXID_shift = ADC_shift + ADC_bits;
  constexpr int DCOL_shift = PXID_shift + PXID_bits;
  constexpr int ROC_shift  = DCOL_shift + DCOL_bits;
//  constexpr int LINK_shift = ROC_shift + ROC_bits;
  constexpr int OMIT_ERR_shift = 20;
 
  constexpr RPixErrorChecker::Word64 CRC_mask = ~(~RPixErrorChecker::Word64(0) << CRC_bits);
  constexpr RPixErrorChecker::Word32 ERROR_mask = ~(~RPixErrorChecker::Word32(0) << ROC_bits);
//  constexpr RPixErrorChecker::Word32 LINK_mask = ~(~RPixErrorChecker::Word32(0) << LINK_bits);
//  constexpr RPixErrorChecker::Word32 ROC_mask  = ~(~RPixErrorChecker::Word32(0) << ROC_bits);
  constexpr RPixErrorChecker::Word32 OMIT_ERR_mask = ~(~RPixErrorChecker::Word32(0) << OMIT_ERR_bits);
}  

RPixErrorChecker::RPixErrorChecker() {

}

bool RPixErrorChecker::checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer)
{
  int CRC_BIT = (*trailer >> CRC_shift) & CRC_mask;
  if (CRC_BIT == 0) return true;
  errorsInEvent = true;
  LogError("CRCCheck")
    <<"CRC check failed,  errorType = 39";
  return false;
}

bool RPixErrorChecker::checkHeader(bool& errorsInEvent, int fedId, const Word64* header)
{
  FEDHeader fedHeader( reinterpret_cast<const unsigned char*>(header));
  if ( !fedHeader.check() ) return false; // throw exception?
  if ( fedHeader.sourceID() != fedId) { 
    LogError("CTPPSPixelDataFormatter::interpretRawData, fedHeader.sourceID() != fedId")
      <<", sourceID = " <<fedHeader.sourceID()
      <<", fedId = "<<fedId<<", errorType = 32"; 
    errorsInEvent = true;

  }
  return fedHeader.moreHeaders();
}

bool RPixErrorChecker::checkTrailer(bool& errorsInEvent, int fedId, int nWords, const Word64* trailer)
{
  FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer));
  if ( !fedTrailer.check()) { 

    errorsInEvent = true;
    LogError("FedTrailerCheck")
      <<"fedTrailer.check failed, Fed: " << fedId << ", errorType = 33";
    return false; 
  } 
  if ( fedTrailer.lenght()!= nWords) {
    LogError("FedTrailerLenght")<< "fedTrailer.lenght()!= nWords !! Fed: " << fedId << ", errorType = 34";
    errorsInEvent = true;
 
  }
  return fedTrailer.moreTrailers();
}

bool RPixErrorChecker::checkROC(bool& errorsInEvent, int fedId,  Word32& errorWord)
{
  int errorType = (errorWord >> ROC_shift) & ERROR_mask;
  if likely(errorType<25) return true;

  switch (errorType) {
  case(25) : {
    LogDebug("")<<"  invalid ROC=25 found (errorType=25)";
    errorsInEvent = true;
    break;
  }
  case(26) : {
  //LogDebug("")<<"  gap word found (errorType=26)";
    return false;
  }
  case(27) : {
  //LogDebug("")<<"  dummy word found (errorType=27)";
    return false;
  }
  case(28) : {
    LogDebug("")<<"  error fifo nearly full (errorType=28)";
    errorsInEvent = true;
    break;
  }
  case(29) : {
    LogDebug("")<<"  timeout on a channel (errorType=29)";
    errorsInEvent = true;
    if ((errorWord >> OMIT_ERR_shift) & OMIT_ERR_mask) {
      LogDebug("")<<"  ...first errorType=29 error, this gets masked out";
      return false;
    }
    break;
  }
  case(30) : {
    LogDebug("")<<"  TBM error trailer (errorType=30)";
    errorsInEvent = true;
    break;
  }
  case(31) : {
    LogDebug("")<<"  event number error (errorType=31)";
    errorsInEvent = true;
    break;
  }
  default: return true;
  };

  return false;
}

