#include "EventFilter/CTPPSRawToDigi/interface/RPixErrorChecker.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <sstream>
#include <iostream>

using namespace std;
using namespace edm;

   constexpr int RPixErrorChecker::CRC_bits ;
   constexpr int RPixErrorChecker::ROC_bits ;
   constexpr int RPixErrorChecker::DCOL_bits;
   constexpr int RPixErrorChecker::PXID_bits;
   constexpr int RPixErrorChecker::ADC_bits ;
   constexpr int RPixErrorChecker::OMIT_ERR_bits;
  
   constexpr int RPixErrorChecker::CRC_shift;
   constexpr int RPixErrorChecker::ADC_shift ;
   constexpr int RPixErrorChecker::PXID_shift;
   constexpr int RPixErrorChecker::DCOL_shift;
   constexpr int RPixErrorChecker::ROC_shift ;
   constexpr int RPixErrorChecker::OMIT_ERR_shift ;
 
   constexpr RPixErrorChecker::Word32 RPixErrorChecker::dummyDetId ;

   constexpr RPixErrorChecker::Word64 RPixErrorChecker::CRC_mask;
   constexpr RPixErrorChecker::Word32 RPixErrorChecker::ERROR_mask ;
   constexpr RPixErrorChecker::Word32 RPixErrorChecker::OMIT_ERR_mask;


RPixErrorChecker::RPixErrorChecker() 
{
  includeErrors = false;
}

void RPixErrorChecker::setErrorStatus(bool ErrorStatus)
{
  includeErrors = ErrorStatus;
}

bool RPixErrorChecker::checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer,  Errors& errors) const
{
  int CRC_BIT = (*trailer >> CRC_shift) & CRC_mask;
  if (CRC_BIT == 0) return true;
  errorsInEvent = true;
  LogError("CRCCheck")
    <<"CRC check failed,  errorType = 39";
  if (includeErrors) {
    int errorType = 39;
    CTPPSPixelDataError error(*trailer, errorType, fedId);
    errors[dummyDetId].push_back(error);
  }
  return false;
}

bool RPixErrorChecker::checkHeader(bool& errorsInEvent, int fedId, const Word64* header,  Errors& errors) const
{
  FEDHeader fedHeader( reinterpret_cast<const unsigned char*>(header));
  if ( !fedHeader.check() ) return false; // throw exception?
  if ( fedHeader.sourceID() != fedId) { 
    LogError("CTPPSPixelDataFormatter::interpretRawData, fedHeader.sourceID() != fedId")
      <<", sourceID = " <<fedHeader.sourceID()
      <<", fedId = "<<fedId<<", errorType = 32"; 
    errorsInEvent = true;
    if (includeErrors) {
      int errorType = 32;
      CTPPSPixelDataError error(*header, errorType, fedId);
      errors[dummyDetId].push_back(error);
    }
  }
  return fedHeader.moreHeaders();
}

bool RPixErrorChecker::checkTrailer(bool& errorsInEvent, int fedId, unsigned int nWords, const Word64* trailer,  Errors& errors) const
{
  FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer));
  if ( !fedTrailer.check()) { 
    if(includeErrors) {
      int errorType = 33;
      CTPPSPixelDataError error(*trailer, errorType, fedId);
      errors[dummyDetId].push_back(error);
    }
    errorsInEvent = true;
    LogError("FedTrailerCheck")
      <<"fedTrailer.check failed, Fed: " << fedId << ", errorType = 33";
    return false; 
  } 
  if ( fedTrailer.fragmentLength()!= nWords) {
    LogError("FedTrailerLenght")<< "fedTrailer.fragmentLength()!= nWords !! Fed: " << fedId << ", errorType = 34";
    errorsInEvent = true;
    if(includeErrors) {
      int errorType = 34;
      CTPPSPixelDataError error(*trailer, errorType, fedId);
      errors[dummyDetId].push_back(error);
    }
  }
  return fedTrailer.moreTrailers();
}

bool RPixErrorChecker::checkROC(bool& errorsInEvent, int fedId, uint32_t iD, Word32& errorWord,  Errors& errors) const
{
  int errorType = (errorWord >> ROC_shift) & ERROR_mask;
  if likely(errorType<25) return true;

  switch (errorType) {
  case(25) : {
    LogError("")<<"  invalid ROC=25 found (errorType=25)";
    errorsInEvent = true;
    break;
  }
  case(26) : {
  //LogError("")<<"  gap word found (errorType=26)";
    return false;
  }
  case(27) : {
  //LogError("")<<"  dummy word found (errorType=27)";
    return false;
  }
  case(28) : {
    LogError("")<<"  error fifo nearly full (errorType=28)";
    errorsInEvent = true;
    break;
  }
  case(29) : {
    LogError("")<<"  timeout on a channel (errorType=29)";
    errorsInEvent = true;
    if ((errorWord >> OMIT_ERR_shift) & OMIT_ERR_mask) {
      LogError("")<<"  ...first errorType=29 error, this gets masked out";
      return false;
    }
    break;
  }
  case(30) : {
    LogError("")<<"  TBM error trailer (errorType=30)";
    errorsInEvent = true;
    break;
  }
  case(31) : {
    LogError("")<<"  event number error (errorType=31)";
    errorsInEvent = true;
    break;
  }
  default: return true;
  };

 if(includeErrors) {
   // check to see if overflow error for type 30, change type to 40 if so
   if(errorType==30) {
     int StateMach_bits      = 4;
     int StateMach_shift     = 8;
     uint32_t StateMach_mask = ~(~uint32_t(0) << StateMach_bits);
     int StateMach = (errorWord >> StateMach_shift) & StateMach_mask;
     if( StateMach==4 || StateMach==9 ) errorType = 40;
   }

   // store error
   CTPPSPixelDataError error(errorWord, errorType, fedId);

   errors[iD].push_back(error);
}

  return false;
}

void RPixErrorChecker::conversionError(int fedId, uint32_t iD, int status, Word32& errorWord, Errors& errors) const
{
  switch (status) {
  case(1) : {
    LogError("ErrorChecker::conversionError") << " Fed: " << fedId << "  invalid channel Id (errorType=35)";
    if(includeErrors) {
      int errorType = 35;
      CTPPSPixelDataError error(errorWord, errorType, fedId);
      errors[iD].push_back(error);
    }
    break;
  }
  case(2) : {
    LogError("ErrorChecker::conversionError")<< " Fed: " << fedId << "  invalid ROC Id (errorType=36)";
    if(includeErrors) {
      int errorType = 36;
      CTPPSPixelDataError error(errorWord, errorType, fedId);
      errors[iD].push_back(error);
    }
    break;
  }
  case(3) : {
    LogError("ErrorChecker::conversionError")<< " Fed: " << fedId << "  invalid dcol/pixel value (errorType=37)";
    if(includeErrors) {
      int errorType = 37;
      CTPPSPixelDataError error(errorWord, errorType, fedId);
      errors[iD].push_back(error);
    }
    break;
  }

  default: LogError("ErrorChecker::conversionError")<<"  cabling check returned unexpected result, status = "<< status;
  };
}
