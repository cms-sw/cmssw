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

void ErrorCheckerBase::addErrorToCollectionDummy(int errorType,
                                                 int fedId,
                                                 Word64 word,
                                                 SiPixelFormatterErrors& errors) const {
  if (includeErrors_) {
    SiPixelRawDataError error(word, errorType, fedId);
    errors[sipixelconstants::dummyDetId].push_back(error);
  }
}

bool ErrorCheckerBase::checkCRC(bool& errorsInEvent,
                                int fedId,
                                const Word64* trailer,
                                SiPixelFormatterErrors& errors) const {
  const int CRC_BIT = (*trailer >> CRC_shift) & CRC_mask;
  const bool isCRCcorrect = (CRC_BIT == 0);
  if (!isCRCcorrect)
    addErrorToCollectionDummy(39, fedId, *trailer, errors);
  errorsInEvent = (errorsInEvent || !isCRCcorrect);
  return isCRCcorrect;
}

bool ErrorCheckerBase::checkHeader(bool& errorsInEvent,
                                   int fedId,
                                   const Word64* header,
                                   SiPixelFormatterErrors& errors) const {
  FEDHeader fedHeader(reinterpret_cast<const unsigned char*>(header));
  const bool fedHeaderCorrect = fedHeader.check();
  // if not fedHeaderCorrect throw exception?
  if (fedHeaderCorrect && (fedHeader.sourceID() != fedId)) {
    int errorType = 32;
    addErrorToCollectionDummy(errorType, fedId, *header, errors);
    LogDebug("PixelDataFormatter::interpretRawData, fedHeader.sourceID() != fedId")
        << ", sourceID = " << fedHeader.sourceID() << ", fedId = " << fedId << ", errorType = " << errorType;
    errorsInEvent = true;
  }
  return fedHeaderCorrect && fedHeader.moreHeaders();
}

bool ErrorCheckerBase::checkTrailer(
    bool& errorsInEvent, int fedId, unsigned int nWords, const Word64* trailer, SiPixelFormatterErrors& errors) const {
  FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer));
  const bool fedTrailerCorrect = fedTrailer.check();
  if (!fedTrailerCorrect) {
    int errorType = 33;
    addErrorToCollectionDummy(errorType, fedId, *trailer, errors);
    LogError("FedTrailerCheck") << "fedTrailer.check failed, Fed: " << fedId << ", errorType = " << errorType;
    errorsInEvent = true;
  } else if (fedTrailer.fragmentLength() != nWords) {
    int errorType = 34;
    addErrorToCollectionDummy(errorType, fedId, *trailer, errors);
    LogError("FedTrailerLenght") << "fedTrailer.fragmentLength()!= nWords !! Fed: " << fedId
                                 << ", errorType = " << errorType;
    errorsInEvent = true;
  }
  return fedTrailerCorrect && fedTrailer.moreTrailers();
}

void ErrorCheckerBase::conversionError(int fedId,
                                       const SiPixelFrameConverter* converter,
                                       int status,
                                       Word32& errorWord,
                                       SiPixelFormatterErrors& errors) const {
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
