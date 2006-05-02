#ifndef DataFormats_CSCRecHit_CSCRangeMapAccessor_H
#define DataFormats_CSCRecHit_CSCRangeMapAccessor_H

/** \class CSCRangeMapAccessor
 *  Comparator to retrieve CSCrechits by chamber.
 *
 *  $Date: 2006/04/03 14:14:12 $
 *  \author Matteo Sani
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCDetIdSameChamberComparator {
public:
  bool operator()(CSCDetId i1, CSCDetId i2) const;
};

class CSCRangeMapAccessor {
public:
  //
  // returns a valid DetId + a valid comaprator for the RangeMap
  //
  CSCRangeMapAccessor();
  std::pair<CSCDetId,CSCDetIdSameChamberComparator> cscChamber(CSCDetId chamber);
  
private:
  
};

#endif
