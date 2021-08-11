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
using namespace sipixelconstants;

ErrorCheckerPhase0::ErrorCheckerPhase0() : ErrorCheckerBase(){};

bool ErrorCheckerPhase0::checkROC(bool& errorsInEvent,
                                  int fedId,
                                  const SiPixelFrameConverter* converter,
                                  const SiPixelFedCabling* theCablingTree,
                                  Word32& errorWord,
                                  SiPixelFormatterErrors& errors) const {
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

  if (includeErrors_) {
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
