#include "EventFilter/CTPPSRawToDigi/interface/RPixErrorChecker.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;

constexpr RPixErrorChecker::Word32 RPixErrorChecker::dummyDetId;

RPixErrorChecker::RPixErrorChecker() { includeErrors_ = false; }

void RPixErrorChecker::setErrorStatus(bool errorStatus) { includeErrors_ = errorStatus; }

bool RPixErrorChecker::checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer, Errors& errors) const {
  int CRC_BIT = (*trailer >> CRC_shift) & CRC_mask;
  if (CRC_BIT == 0)
    return true;
  errorsInEvent = true;
  LogDebug("CRCCheck") << "CRC check failed,  errorType = 39";
  if (includeErrors_) {
    int errorType = 39;
    errors[dummyDetId].emplace_back(*trailer, errorType, fedId);
  }
  return false;
}

bool RPixErrorChecker::checkHeader(bool& errorsInEvent, int fedId, const Word64* header, Errors& errors) const {
  FEDHeader fedHeader(reinterpret_cast<const unsigned char*>(header));
  if (!fedHeader.check())
    return false;
  if (fedHeader.sourceID() != fedId) {
    LogDebug("CTPPSPixelDataFormatter::interpretRawData, fedHeader.sourceID() != fedId")
        << ", sourceID = " << fedHeader.sourceID() << ", fedId = " << fedId << ", errorType = 32";
    errorsInEvent = true;
    if (includeErrors_) {
      int errorType = 32;
      errors[dummyDetId].emplace_back(*header, errorType, fedId);
    }
  }
  return fedHeader.moreHeaders();
}

bool RPixErrorChecker::checkTrailer(
    bool& errorsInEvent, int fedId, unsigned int nWords, const Word64* trailer, Errors& errors) const {
  FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer));
  if (!fedTrailer.check()) {
    if (includeErrors_) {
      int errorType = 33;
      errors[dummyDetId].emplace_back(*trailer, errorType, fedId);
    }
    errorsInEvent = true;
    LogDebug("FedTrailerCheck") << "fedTrailer.check failed, Fed: " << fedId << ", errorType = 33";
    return false;
  }
  if (fedTrailer.fragmentLength() != nWords) {
    LogDebug("FedTrailerLenght") << "fedTrailer.fragmentLength()!= nWords !! Fed: " << fedId << ", errorType = 34";
    errorsInEvent = true;
    if (includeErrors_) {
      int errorType = 34;
      errors[dummyDetId].emplace_back(*trailer, errorType, fedId);
    }
  }
  return fedTrailer.moreTrailers();
}

bool RPixErrorChecker::checkROC(
    bool& errorsInEvent, int fedId, uint32_t iD, const Word32& errorWord, Errors& errors) const {
  int errorType = (errorWord >> ROC_shift) & ERROR_mask;
  if LIKELY (errorType < 25)
    return true;

  switch (errorType) {
    case (25): {
      LogDebug("") << "  invalid ROC=25 found (errorType=25)";
      errorsInEvent = true;
      break;
    }
    case (26): {
      LogDebug("") << "  gap word found (errorType=26)";
      return false;
    }
    case (27): {
      LogDebug("") << "  dummy word found (errorType=27)";
      return false;
    }
    case (28): {
      LogDebug("") << "  error fifo nearly full (errorType=28)";
      errorsInEvent = true;
      break;
    }
    case (29): {
      LogDebug("") << "  timeout on a channel (errorType=29)";
      errorsInEvent = true;
      if ((errorWord >> OMIT_ERR_shift) & OMIT_ERR_mask) {
        LogDebug("") << "  ...first errorType=29 error, this gets masked out";
        return false;
      }
      break;
    }
    case (30): {
      LogDebug("") << "  TBM error trailer (errorType=30)";
      errorsInEvent = true;
      break;
    }
    case (31): {
      LogDebug("") << "  event number error (errorType=31)";
      errorsInEvent = true;
      break;
    }
    default:
      return true;
  };

  if (includeErrors_) {
    /// check to see if overflow error for type 30, change type to 40 if so
    if (errorType == 30) {
      uint32_t stateMach_bits = 4;
      uint32_t stateMach_shift = 8;
      uint32_t stateMach_mask = ~(~uint32_t(0) << stateMach_bits);
      uint32_t stateMach = (errorWord >> stateMach_shift) & stateMach_mask;
      if (stateMach == 4 || stateMach == 9)
        errorType = 40;
    }

    /// store error
    errors[iD].emplace_back(errorWord, errorType, fedId);
  }

  return false;
}

void RPixErrorChecker::conversionError(
    int fedId, uint32_t iD, const State& state, const Word32& errorWord, Errors& errors) const {
  int errorType = 0;

  switch (state) {
    case (InvalidLinkId): {
      LogDebug("ErrorChecker::conversionError") << " Fed: " << fedId << "  invalid channel Id (errorType=35)";
      errorType = 35;
      break;
    }
    case (InvalidROCId): {
      LogDebug("ErrorChecker::conversionError") << " Fed: " << fedId << "  invalid ROC Id (errorType=36)";
      errorType = 36;
      break;
    }
    case (InvalidPixelId): {
      LogDebug("ErrorChecker::conversionError") << " Fed: " << fedId << "  invalid dcol/pixel value (errorType=37)";
      errorType = 37;
      break;
    }

    default:
      LogDebug("ErrorChecker::conversionError") << "  cabling check returned unexpected result, status = " << state;
  };

  if (includeErrors_ && errorType > 0)
    errors[iD].emplace_back(errorWord, errorType, fedId);
}
