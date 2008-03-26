#include "EventFilter/SiPixelRawToDigi/interface/ErrorChecker.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <sstream>
#include <iostream>

using namespace std;
using namespace edm;
using namespace sipixelobjects;

const int ErrorChecker::LINK_bits = 6;
const int ErrorChecker::ROC_bits  = 5;
const int ErrorChecker::DCOL_bits = 5;
const int ErrorChecker::PXID_bits = 8;
const int ErrorChecker::ADC_bits  = 8;

const int ErrorChecker::ADC_shift  = 0;
const int ErrorChecker::PXID_shift = ADC_shift + ADC_bits;
const int ErrorChecker::DCOL_shift = PXID_shift + PXID_bits;
const int ErrorChecker::ROC_shift  = DCOL_shift + DCOL_bits;
const int ErrorChecker::LINK_shift = ROC_shift + ROC_bits;

const uint32_t ErrorChecker::dummyDetId = 0xffffffff;


ErrorChecker::ErrorChecker() {
  includeErrors = false;
}

void ErrorChecker::setErrorStatus(bool ErrorStatus)
{
  includeErrors = ErrorStatus;
}

bool ErrorChecker::checkHeader(int fedId, const Word64* header, Errors& errors)
{
  FEDHeader fedHeader( reinterpret_cast<const unsigned char*>(header));
  if ( !fedHeader.check() ) return false; // throw exception?
  if ( fedHeader.sourceID() != fedId) { 
    LogError("PixelDataFormatter::interpretRawData, fedHeader.sourceID() != fedId")
      <<", sourceID = " <<fedHeader.sourceID()
      <<", fedId = "<<fedId<<", errorType = 32"; 
    if (includeErrors) {
      int errorType = 32;
      SiPixelRawDataError error(*header, errorType, fedId);
      errors[dummyDetId].push_back(error);
    }
  }
  return fedHeader.moreHeaders();
}

bool ErrorChecker::checkTrailer(int fedId, int nWords, const Word64* trailer, Errors& errors)
{
  FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer));
  if ( !fedTrailer.check()) { 
    if(includeErrors) {
      int errorType = 33;
      SiPixelRawDataError error(*trailer, errorType, fedId);
      errors[dummyDetId].push_back(error);
    }
    LogError("PixelDataFormatter::interpretRawData, fedTrailer.check: ")
      <<"fedTrailer.check failed"<<", errorType = 33";
    return false; 
  } 
  if ( fedTrailer.lenght()!= nWords) {
    LogError("PROBLEM in PixelDataFormatter,  fedTrailer.lenght()!= nWords !!")<<", errorType = 34";
    if(includeErrors) {
      int errorType = 34;
      SiPixelRawDataError error(*trailer, errorType, fedId);
      errors[dummyDetId].push_back(error);
    }
  }
  return fedTrailer.moreTrailers();
}

bool ErrorChecker::checkROC(int fedId, const SiPixelFrameConverter* converter, Word32 errorWord, Errors& errors)
{
 static const Word32 ERROR_mask = ~(~Word32(0) << ROC_bits); 
 int errorType = (errorWord >> ROC_shift) & ERROR_mask;

 switch (errorType) {
    case(25) : {
     LogTrace("")<<"  invalid ROC=25 found (errorType=25)";
     break;
   }
   case(26) : {
     LogTrace("")<<"  gap word found (errorType=26)";
     break;
   }
   case(27) : {
     LogTrace("")<<"  dummy word found (errorType=27)";
     break;
   }
   case(28) : {
     LogTrace("")<<"  error fifo nearly full (errorType=28)";
     break;
   }
   case(29) : {
     LogTrace("")<<"  timeout on a channel (errorType=29)";
     break;
   }
   case(30) : {
     LogTrace("")<<"  trailer error (errorType=30)";
     break;
   }
   case(31) : {
     LogTrace("")<<"  event number error (errorType=31)";
     break;
   }
   default: return true;
 };

 if(includeErrors) {
   SiPixelRawDataError error(errorWord, errorType, fedId);
   uint32_t detId;
   detId = errorDetId(converter, errorType, errorWord);
   errors[detId].push_back(error);
 }
 return false;
}

void ErrorChecker::conversionError(int fedId, const SiPixelFrameConverter* converter, int status, Word32 errorWord, Errors& errors)
{
  switch (status) {
  case(1) : {
    LogError("PixelDataFormatter::interpretRawData")<<"  invalid channel Id (errorType=35)";
    if(includeErrors) {
      int errorType = 35;
      SiPixelRawDataError error(errorWord, errorType, fedId);
      uint32_t detId = errorDetId(converter, errorType, errorWord);
      errors[detId].push_back(error);
    }
    break;
  }
  case(2) : {
    LogError("PixelDataFormatter::interpretRawData")<<"  invalid ROC Id (errorType=36)";
    if(includeErrors) {
      int errorType = 36;
      SiPixelRawDataError error(errorWord, errorType, fedId);
      uint32_t detId = errorDetId(converter, errorType, errorWord);
      errors[detId].push_back(error);
    }
    break;
  }
  case(3) : {
    LogError("PixelDataFormatter::interpretRawData")<<"  invalid dcol/pixel value (errorType=37)";
    if(includeErrors) {
      int errorType = 37;
      SiPixelRawDataError error(errorWord, errorType, fedId);
      uint32_t detId = errorDetId(converter, errorType, errorWord);
      errors[detId].push_back(error);
    }
    break;
  }
  case(4) : {
    LogError("PixelDataFormatter::interpretRawData")<<"  dcol/pixel read out of order (errorType=38)";
    if(includeErrors) {
      int errorType = 38;
      SiPixelRawDataError error(errorWord, errorType, fedId);
      uint32_t detId = errorDetId(converter, errorType, errorWord);
      errors[detId].push_back(error);
    }
    break;
  }
  default: LogError("PixelDataFormatter::interpretRawData")<<"  cabling check returned unexpected result";
  };
}

// this function finds the detId for an error word, which cannot be processed in word2digi
uint32_t ErrorChecker::errorDetId(const SiPixelFrameConverter* converter, 
    int errorType, const Word32 & word) const
{
  if (!converter) return dummyDetId;

  ElectronicIndex cabling;

  static const Word32 LINK_mask = ~(~Word32(0) << LINK_bits);
  static const Word32 ROC_mask  = ~(~Word32(0) << ROC_bits);
  static const Word32 DCOL_mask = ~(~Word32(0) << DCOL_bits);
  static const Word32 PXID_mask = ~(~Word32(0) << PXID_bits);

  switch (errorType) {
    case  30 : case  31: case  36: {
      // set dummy values for cabling just to get detId from link if in Barrel
      cabling.dcol = 0;
      cabling.pxid = 2;
      cabling.roc  = 1;
      cabling.link = (word >> LINK_shift) & LINK_mask;  

      DetectorIndex detIdx;
      int status = converter->toDetector(cabling, detIdx);
      
      if(DetId::DetId(detIdx.rawId).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) return detIdx.rawId;
      break;
    }
    case  37 : case  38: {
      cabling.dcol = 0;
      cabling.pxid = 2;
      cabling.roc  = (word >> ROC_shift) & ROC_mask;
      cabling.link = (word >> LINK_shift) & LINK_mask;

      DetectorIndex detIdx;
      int status = converter->toDetector(cabling, detIdx);

      return detIdx.rawId;
      break;
    }
  default : break;
  };
  return dummyDetId;
}
