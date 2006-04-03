#ifndef DataFormats_CSCDetIdAccessor_H
#define DataFormats_CSCDetIdAccessor_H

/** \class CSCChamberIdComparator
 *  Comparator to retrieve CSCrechits by chamber.
 *
 *  $Date: 2006/03/31 16:48:37 $
 *  \author Matteo Sani
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCDetIdSameChamberComparator {
public:
  bool operator()(CSCDetId i1, CSCDetId i2) const;
};

class CSCDetIdAccessor {
public:
  //
  // returns a valid DetId + a valid comaprator for the RangeMap
  //
  CSCDetIdAccessor();
  std::pair<CSCDetId,CSCDetIdSameChamberComparator> cscChamber(CSCDetId chamber);
  
private:
  
};

#endif
