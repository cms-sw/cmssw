#include "EventFilter/SiPixelRawToDigi/interface/ErrorCheckerPhase0.h"

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

  constexpr ErrorCheckerPhase0::Word64 CRC_mask = ~(~ErrorCheckerPhase0::Word64(0) << CRC_bits);
  constexpr ErrorCheckerPhase0::Word32 ERROR_mask = ~(~ErrorCheckerPhase0::Word32(0) << ROC_bits);
  constexpr ErrorCheckerPhase0::Word32 LINK_mask = ~(~ErrorCheckerPhase0::Word32(0) << LINK_bits);
  constexpr ErrorCheckerPhase0::Word32 ROC_mask = ~(~ErrorCheckerPhase0::Word32(0) << ROC_bits);
  constexpr ErrorCheckerPhase0::Word32 OMIT_ERR_mask = ~(~ErrorCheckerPhase0::Word32(0) << OMIT_ERR_bits);
}  // namespace

ErrorCheckerPhase0::ErrorCheckerPhase0() { includeErrors = false; }

void ErrorCheckerPhase0::setErrorStatus(bool ErrorStatus) { includeErrors = ErrorStatus; }

bool ErrorCheckerPhase0::checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer, Errors& errors) {
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

bool ErrorCheckerPhase0::checkHeader(bool& errorsInEvent, int fedId, const Word64* header, Errors& errors) {
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

bool ErrorCheckerPhase0::checkTrailer(
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

bool ErrorCheckerPhase0::checkROC(bool& errorsInEvent,
                                  int fedId,
                                  const SiPixelFrameConverter* converter,
                                  const SiPixelFedCabling* theCablingTree,
                                  Word32& errorWord,
                                  Errors& errors) {
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
      return false;
    }
    case (27): {
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

  if (includeErrors) {
    // check to see if overflow error for type 30, change type to 40 if so
    if (errorType == 30) {
      int StateMach_bits = 4;
      int StateMach_shift = 8;
      uint32_t StateMach_mask = ~(~uint32_t(0) << StateMach_bits);
      int StateMach = (errorWord >> StateMach_shift) & StateMach_mask;
      if (StateMach == 4 || StateMach == 9)
        errorType = 40;
    }

    // store error
    SiPixelRawDataError error(errorWord, errorType, fedId);
    cms_uint32_t detId;
    detId = errorDetId(converter, errorType, errorWord);
    errors[detId].push_back(error);
  }
  return false;
}

void ErrorCheckerPhase0::conversionError(
    int fedId, const SiPixelFrameConverter* converter, int status, Word32& errorWord, Errors& errors) {
  switch (status) {
    case (1): {
      LogDebug("ErrorCheckerPhase0::conversionError") << " Fed: " << fedId << "  invalid channel Id (errorType=35)";
      if (includeErrors) {
        int errorType = 35;
        SiPixelRawDataError error(errorWord, errorType, fedId);
        cms_uint32_t detId = errorDetId(converter, errorType, errorWord);
        errors[detId].push_back(error);
      }
      break;
    }
    case (2): {
      LogDebug("ErrorCheckerPhase0::conversionError") << " Fed: " << fedId << "  invalid ROC Id (errorType=36)";
      if (includeErrors) {
        int errorType = 36;
        SiPixelRawDataError error(errorWord, errorType, fedId);
        cms_uint32_t detId = errorDetId(converter, errorType, errorWord);
        errors[detId].push_back(error);
      }
      break;
    }
    case (3): {
      LogDebug("ErrorCheckerPhase0::conversionError")
          << " Fed: " << fedId << "  invalid dcol/pixel value (errorType=37)";
      if (includeErrors) {
        int errorType = 37;
        SiPixelRawDataError error(errorWord, errorType, fedId);
        cms_uint32_t detId = errorDetId(converter, errorType, errorWord);
        errors[detId].push_back(error);
      }
      break;
    }
    case (4): {
      LogDebug("ErrorCheckerPhase0::conversionError")
          << " Fed: " << fedId << "  dcol/pixel read out of order (errorType=38)";
      if (includeErrors) {
        int errorType = 38;
        SiPixelRawDataError error(errorWord, errorType, fedId);
        cms_uint32_t detId = errorDetId(converter, errorType, errorWord);
        errors[detId].push_back(error);
      }
      break;
    }
    default:
      LogDebug("ErrorCheckerPhase0::conversionError")
          << "  cabling check returned unexpected result, status = " << status;
  };
}

// this function finds the detId for an error word that cannot be processed in word2digi
cms_uint32_t ErrorCheckerPhase0::errorDetId(const SiPixelFrameConverter* converter,
                                            int errorType,
                                            const Word32& word) const {
  if (!converter)
    return dummyDetId;

  ElectronicIndex cabling;

  switch (errorType) {
    case 25:
    case 30:
    case 31:
    case 36:
    case 40: {
      // set dummy values for cabling just to get detId from link if in Barrel
      cabling.dcol = 0;
      cabling.pxid = 2;
      cabling.roc = 1;
      cabling.link = (word >> LINK_shift) & LINK_mask;

      DetectorIndex detIdx;
      int status = converter->toDetector(cabling, detIdx);
      if (status)
        break;
      if (DetId(detIdx.rawId).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel))
        return detIdx.rawId;
      break;
    }
    case 29: {
      int chanNmbr = 0;
      const int DB0_shift = 0;
      const int DB1_shift = DB0_shift + 1;
      const int DB2_shift = DB1_shift + 1;
      const int DB3_shift = DB2_shift + 1;
      const int DB4_shift = DB3_shift + 1;
      const cms_uint32_t DataBit_mask = ~(~cms_uint32_t(0) << 1);

      int CH1 = (word >> DB0_shift) & DataBit_mask;
      int CH2 = (word >> DB1_shift) & DataBit_mask;
      int CH3 = (word >> DB2_shift) & DataBit_mask;
      int CH4 = (word >> DB3_shift) & DataBit_mask;
      int CH5 = (word >> DB4_shift) & DataBit_mask;
      int BLOCK_bits = 3;
      int BLOCK_shift = 8;
      cms_uint32_t BLOCK_mask = ~(~cms_uint32_t(0) << BLOCK_bits);
      int BLOCK = (word >> BLOCK_shift) & BLOCK_mask;
      int localCH = 1 * CH1 + 2 * CH2 + 3 * CH3 + 4 * CH4 + 5 * CH5;
      if (BLOCK % 2 == 0)
        chanNmbr = (BLOCK / 2) * 9 + localCH;
      else
        chanNmbr = ((BLOCK - 1) / 2) * 9 + 4 + localCH;
      if ((chanNmbr < 1) || (chanNmbr > 36))
        break;  // signifies unexpected result

      // set dummy values for cabling just to get detId from link if in Barrel
      cabling.dcol = 0;
      cabling.pxid = 2;
      cabling.roc = 1;
      cabling.link = chanNmbr;
      DetectorIndex detIdx;
      int status = converter->toDetector(cabling, detIdx);
      if (status)
        break;
      if (DetId(detIdx.rawId).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel))
        return detIdx.rawId;
      break;
    }
    case 37:
    case 38: {
      cabling.dcol = 0;
      cabling.pxid = 2;
      cabling.roc = (word >> ROC_shift) & ROC_mask;
      cabling.link = (word >> LINK_shift) & LINK_mask;

      DetectorIndex detIdx;
      int status = converter->toDetector(cabling, detIdx);
      if (status)
        break;

      return detIdx.rawId;
      break;
    }
    default:
      break;
  };
  return dummyDetId;
}
