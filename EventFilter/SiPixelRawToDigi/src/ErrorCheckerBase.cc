#include "EventFilter/SiPixelRawToDigi/interface/ErrorCheckerBase.h"

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

namespace {
  constexpr int CRC_bits = 1;
  constexpr int LINK_bits = 6;
  constexpr int ROC_bits = 5;
  constexpr int DCOL_bits = 5;
  constexpr int PXID_bits = 8;
  constexpr int ADC_bits = 8;
  constexpr int OMIT_ERR_bits = 1;

  constexpr int CRC_shift = 2;
  constexpr int ADC_shift = 0;
  constexpr int PXID_shift = ADC_shift + ADC_bits;
  constexpr int DCOL_shift = PXID_shift + PXID_bits;
  constexpr int ROC_shift = DCOL_shift + DCOL_bits;
  constexpr int LINK_shift = ROC_shift + ROC_bits;
  constexpr int OMIT_ERR_shift = 20;

  constexpr cms_uint32_t dummyDetId = 0xffffffff;

  constexpr ErrorCheckerBase::Word64 CRC_mask = ~(~ErrorCheckerBase::Word64(0) << CRC_bits);
  constexpr ErrorCheckerBase::Word32 ERROR_mask = ~(~ErrorCheckerBase::Word32(0) << ROC_bits);
  constexpr ErrorCheckerBase::Word32 LINK_mask = ~(~ErrorCheckerBase::Word32(0) << LINK_bits);
  constexpr ErrorCheckerBase::Word32 ROC_mask = ~(~ErrorCheckerBase::Word32(0) << ROC_bits);
  constexpr ErrorCheckerBase::Word32 OMIT_ERR_mask = ~(~ErrorCheckerBase::Word32(0) << OMIT_ERR_bits);
}  // namespace

ErrorCheckerBase::ErrorCheckerBase() : includeErrors(false) {}

void ErrorCheckerBase::setErrorStatus(bool ErrorStatus) { includeErrors = ErrorStatus; }

bool ErrorCheckerBase::checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer, Errors& errors) {
  int CRC_BIT = (*trailer >> CRC_shift) & CRC_mask;
  if (CRC_BIT == 0)
    return true;
  errorsInEvent = true;
  if (includeErrors) {
    int errorType = 39;
    SiPixelRawDataError error(*trailer, errorType, fedId);
    errors[dummyDetId].push_back(error);
  }
  return false;
}

bool ErrorCheckerBase::checkHeader(bool& errorsInEvent, int fedId, const Word64* header, Errors& errors) {
  FEDHeader fedHeader(reinterpret_cast<const unsigned char*>(header));
  if (!fedHeader.check())
    return false;  // throw exception?
  if (fedHeader.sourceID() != fedId) {
    LogDebug("PixelDataFormatter::interpretRawData, fedHeader.sourceID() != fedId")
        << ", sourceID = " << fedHeader.sourceID() << ", fedId = " << fedId << ", errorType = 32";
    errorsInEvent = true;
    if (includeErrors) {
      int errorType = 32;
      SiPixelRawDataError error(*header, errorType, fedId);
      errors[dummyDetId].push_back(error);
    }
  }
  return fedHeader.moreHeaders();
}

bool ErrorCheckerBase::checkTrailer(
    bool& errorsInEvent, int fedId, unsigned int nWords, const Word64* trailer, Errors& errors) {
  FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer));
  if (!fedTrailer.check()) {
    if (includeErrors) {
      int errorType = 33;
      SiPixelRawDataError error(*trailer, errorType, fedId);
      errors[dummyDetId].push_back(error);
    }
    errorsInEvent = true;
    LogError("FedTrailerCheck") << "fedTrailer.check failed, Fed: " << fedId << ", errorType = 33";
    return false;
  }
  if (fedTrailer.fragmentLength() != nWords) {
    LogError("FedTrailerLenght") << "fedTrailer.fragmentLength()!= nWords !! Fed: " << fedId << ", errorType = 34";
    errorsInEvent = true;
    if (includeErrors) {
      int errorType = 34;
      SiPixelRawDataError error(*trailer, errorType, fedId);
      errors[dummyDetId].push_back(error);
    }
  }
  return fedTrailer.moreTrailers();
}

void ErrorCheckerBase::conversionError(
    int fedId, const SiPixelFrameConverter* converter, int status, Word32& errorWord, Errors& errors) {
  switch (status) {
    case (1): {
      LogDebug("ErrorChecker::conversionError") << " Fed: " << fedId << "  invalid channel Id (errorType=35)";
      if (includeErrors) {
        int errorType = 35;
        SiPixelRawDataError error(errorWord, errorType, fedId);
        cms_uint32_t detId = errorDetId(converter, errorType, errorWord);
        errors[detId].push_back(error);
      }
      break;
    }
    case (2): {
      LogDebug("ErrorChecker::conversionError") << " Fed: " << fedId << "  invalid ROC Id (errorType=36)";
      if (includeErrors) {
        int errorType = 36;
        SiPixelRawDataError error(errorWord, errorType, fedId);
        cms_uint32_t detId = errorDetId(converter, errorType, errorWord);
        errors[detId].push_back(error);
      }
      break;
    }
    case (3): {
      LogDebug("ErrorChecker::conversionError") << " Fed: " << fedId << "  invalid dcol/pixel value (errorType=37)";
      if (includeErrors) {
        int errorType = 37;
        SiPixelRawDataError error(errorWord, errorType, fedId);
        cms_uint32_t detId = errorDetId(converter, errorType, errorWord);
        errors[detId].push_back(error);
      }
      break;
    }
    case (4): {
      LogDebug("ErrorChecker::conversionError") << " Fed: " << fedId << "  dcol/pixel read out of order (errorType=38)";
      if (includeErrors) {
        int errorType = 38;
        SiPixelRawDataError error(errorWord, errorType, fedId);
        cms_uint32_t detId = errorDetId(converter, errorType, errorWord);
        errors[detId].push_back(error);
      }
      break;
    }
    default:
      LogDebug("ErrorChecker::conversionError") << "  cabling check returned unexpected result, status = " << status;
  };
}
