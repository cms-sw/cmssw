#ifndef DataFormats_CSCRecHit_CSCRangeMapAccessor_H
#define DataFormats_CSCRecHit_CSCRangeMapAccessor_H

/** \class CSCRangeMapAccessor
 *  Comparator to retrieve CSCrechits by chamber. 
 *
 *  $Date: 2006/05/02 10:39:53 $
 *  \author Matteo Sani
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCDetIdSameChamberComparator {
public:
  bool operator()(CSCDetId i1, CSCDetId i2) const;
};

class CSCRangeMapAccessor {
public:

  /// Constructor
  CSCRangeMapAccessor();
  
  /// Returns a valid DetId + a valid comparator for the RangeMap.
  std::pair<CSCDetId,CSCDetIdSameChamberComparator> cscChamber(CSCDetId chamber);
  
private:
  
};

#endif
