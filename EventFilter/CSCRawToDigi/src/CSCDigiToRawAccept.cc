#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDigiToRawAccept.h"

CSCDetId CSCDigiToRawAccept::chamberID(const CSCDetId& cscDetId) {
  CSCDetId chamberId = cscDetId.chamberId();
  if (chamberId.ring() == 4) {
    chamberId = CSCDetId(chamberId.endcap(), chamberId.station(), 1, chamberId.chamber(), 0);
  }
  return chamberId;
}

// older function for pretriggers BX objects
bool CSCDigiToRawAccept::accept(const CSCDetId& cscId,
                                const CSCCLCTPreTriggerCollection& lcts,
                                int bxMin,
                                int bxMax,
                                int nominalBX,
                                bool me1abCheck) {
  if (bxMin == -999)
    return true;
  CSCDetId chamberId = chamberID(cscId);
  CSCCLCTPreTriggerCollection::Range lctRange = lcts.get(chamberId);
  bool result = false;
  for (CSCCLCTPreTriggerCollection::const_iterator lctItr = lctRange.first; lctItr != lctRange.second; ++lctItr) {
    int bx = *lctItr - nominalBX;
    if (bx >= bxMin && bx <= bxMax) {
      result = true;
      break;
    }
  }
  bool me1a = cscId.station() == 1 && cscId.ring() == 4;
  if (me1a && result == false && me1abCheck) {
    //check pretriggers in me1a as well; relevant for TMB emulator writing to separate detIds
    lctRange = lcts.get(cscId);
    for (CSCCLCTPreTriggerCollection::const_iterator lctItr = lctRange.first; lctItr != lctRange.second; ++lctItr) {
      int bx = *lctItr - nominalBX;
      if (bx >= bxMin && bx <= bxMax) {
        result = true;
        break;
      }
    }
  }
  return result;
}

bool CSCDigiToRawAccept::accept(const CSCDetId& cscId,
                                const CSCCLCTPreTriggerDigiCollection& lcts,
                                int bxMin,
                                int bxMax,
                                int nominalBX,
                                bool me1abCheck,
                                std::vector<bool>& preTriggerInCFEB) {
  if (bxMin == -999)
    return true;

  bool atLeastOnePreTrigger = false;

  CSCDetId chamberId = chamberID(cscId);
  CSCCLCTPreTriggerDigiCollection::Range lctRange = lcts.get(chamberId);
  for (CSCCLCTPreTriggerDigiCollection::const_iterator lctItr = lctRange.first; lctItr != lctRange.second; ++lctItr) {
    int bx = lctItr->getBX() - nominalBX;
    if (bx >= bxMin && bx <= bxMax) {
      atLeastOnePreTrigger = true;
      // save the location of all pretriggers
      preTriggerInCFEB[lctItr->getCFEB()] = true;
    }
  }
  bool me1a = cscId.station() == 1 && cscId.ring() == 4;
  if (me1a && !atLeastOnePreTrigger && me1abCheck) {
    //check pretriggers in me1a as well; relevant for TMB emulator writing to separate detIds
    lctRange = lcts.get(cscId);
    for (CSCCLCTPreTriggerDigiCollection::const_iterator lctItr = lctRange.first; lctItr != lctRange.second; ++lctItr) {
      int bx = lctItr->getBX() - nominalBX;
      if (bx >= bxMin && bx <= bxMax) {
        atLeastOnePreTrigger = true;
        // save the location of all pretriggers
        preTriggerInCFEB[lctItr->getCFEB()] = true;
      }
    }
  }
  return atLeastOnePreTrigger;
}
