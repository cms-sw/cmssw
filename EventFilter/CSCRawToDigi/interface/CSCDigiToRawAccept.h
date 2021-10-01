#ifndef EventFilter_CSCRawToDigi_CSCDigiToRawAccept_h
#define EventFilter_CSCRawToDigi_CSCDigiToRawAccept_h

/** \class CSCDigiToRawAccept
 *
 * Static class with conditions to accept CSC digis in the Digi-to-Raw step
 *
 *  \author Sven Dildick - Rice
 */

#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerDigiCollection.h"

namespace CSCDigiToRawAccept {

  // takes layer ID, converts to chamber ID, switching ME1A to ME11
  CSCDetId chamberID(const CSCDetId& cscDetId);

  /* Was there a trigger primitive in the BX range between bxMin and bxMax?
     The nominalBX argument is 3 for ALCTs and 7 for CLCTs. This is subtracted
     from the object BX before we check if it is in the BX range.

     Triggers in ME1/1 should always be assigned with ring number equal to 1.
     Distinguishing CLCTs in ME1/a and ME1/b is done with the CLCT half-strip,
     or CLCT CFEB.
  */
  template <typename LCTCollection>
  bool accept(const CSCDetId& cscId, const LCTCollection& lcts, int bxMin, int bxMax, int nominalBX) {
    if (bxMin == -999)
      return true;
    CSCDetId chamberId = chamberID(cscId);
    typename LCTCollection::Range lctRange = lcts.get(chamberId);
    bool result = false;
    for (typename LCTCollection::const_iterator lctItr = lctRange.first; lctItr != lctRange.second; ++lctItr) {
      int bx = lctItr->getBX() - nominalBX;
      if (bx >= bxMin && bx <= bxMax) {
        result = true;
        break;
      }
    }
    return result;
  }

  // older implementation for CLCT pretrigger objects that only have BX information
  bool accept(const CSCDetId& cscId, const CSCCLCTPreTriggerCollection& lcts, int bxMin, int bxMax, int nominalBX);

  // newer implementation for CLCT pretrigger objects that have BX and CFEB information
  bool accept(const CSCDetId& cscId,
              const CSCCLCTPreTriggerDigiCollection& lcts,
              int bxMin,
              int bxMax,
              int nominalBX,
              std::vector<bool>& preTriggerInCFEB);
};  // namespace CSCDigiToRawAccept

#endif
