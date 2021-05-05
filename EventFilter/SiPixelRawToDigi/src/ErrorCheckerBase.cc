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
using namespace sipixelconstants;

ErrorCheckerBase::ErrorCheckerBase() : includeErrors_(false) {}

void ErrorCheckerBase::setErrorStatus(bool ErrorStatus) { includeErrors_ = ErrorStatus; }

bool ErrorCheckerBase::checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer, SiPixelFormatterErrors& errors) {
  const int CRC_BIT = (*trailer >> CRC_shift) & CRC_mask;
  const bool isCRCcorrect = (CRC_BIT == 0);
  errorsInEvent = (errorsInEvent || !isCRCcorrect);
  if (includeErrors_ && !isCRCcorrect) {
    const int errorType = 39;
    SiPixelRawDataError error(*trailer, errorType, fedId);
    errors[sipixelconstants::dummyDetId].push_back(error);
  }
  return isCRCcorrect;
}

bool ErrorCheckerBase::checkHeader(bool& errorsInEvent,
                                   int fedId,
                                   const Word64* header,
                                   SiPixelFormatterErrors& errors) {
  FEDHeader fedHeader(reinterpret_cast<const unsigned char*>(header));
  if (!fedHeader.check())
    return false;  // throw exception?
  if (fedHeader.sourceID() != fedId) {
    LogDebug("PixelDataFormatter::interpretRawData, fedHeader.sourceID() != fedId")
        << ", sourceID = " << fedHeader.sourceID() << ", fedId = " << fedId << ", errorType = 32";
    errorsInEvent = true;
    if (includeErrors_) {
      int errorType = 32;
      SiPixelRawDataError error(*header, errorType, fedId);
      errors[sipixelconstants::dummyDetId].push_back(error);
    }
  }
  return fedHeader.moreHeaders();
}

bool ErrorCheckerBase::checkTrailer(
    bool& errorsInEvent, int fedId, unsigned int nWords, const Word64* trailer, SiPixelFormatterErrors& errors) {
  FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer));
  if (!fedTrailer.check()) {
    if (includeErrors_) {
      int errorType = 33;
      SiPixelRawDataError error(*trailer, errorType, fedId);
      errors[sipixelconstants::dummyDetId].push_back(error);
    }
    errorsInEvent = true;
    LogError("FedTrailerCheck") << "fedTrailer.check failed, Fed: " << fedId << ", errorType = 33";
    return false;
  }
  if (fedTrailer.fragmentLength() != nWords) {
    LogError("FedTrailerLenght") << "fedTrailer.fragmentLength()!= nWords !! Fed: " << fedId << ", errorType = 34";
    errorsInEvent = true;
    if (includeErrors_) {
      int errorType = 34;
      SiPixelRawDataError error(*trailer, errorType, fedId);
      errors[sipixelconstants::dummyDetId].push_back(error);
    }
  }
  return fedTrailer.moreTrailers();
}

void ErrorCheckerBase::conversionError(
    int fedId, const SiPixelFrameConverter* converter, int status, Word32& errorWord, SiPixelFormatterErrors& errors) {
  const int errorType = getConversionErrorTypeAndIssueLogMessage(status, fedId);
  // errorType == 0 means unexpected error, in this case we don't include it in the error collection
  if (errorType && includeErrors_) {
    SiPixelRawDataError error(errorWord, errorType, fedId);
    cms_uint32_t detId = errorDetId(converter, errorType, errorWord);
    errors[detId].push_back(error);
  }
}

int ErrorCheckerBase::getConversionErrorTypeAndIssueLogMessage(int status, int fedId) const {
  int errorType = 0;
  std::string debugMessage;
  switch (status) {
    case (1): {
      debugMessage = "invalid channel Id";
      errorType = 35;
      break;
    }
    case (2): {
      debugMessage = "invalid ROC Id";
      errorType = 36;
      break;
    }
    case (3): {
      debugMessage = "invalid dcol/pixel value";
      errorType = 37;
      break;
    }
    case (4): {
      debugMessage = "dcol/pixel read out of order";
      errorType = 38;
      break;
    }
  };
  if (errorType) {
    LogDebug("ErrorChecker::conversionError") << "Fed:" << fedId << debugMessage << "(errorType =" << errorType << ")";
  } else {
    LogDebug("ErrorChecker::conversionError") << "cabling check returned unexpected result, status =" << status;
  }
  return errorType;
}
