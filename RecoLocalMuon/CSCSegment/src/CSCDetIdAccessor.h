#ifndef DataFormats_CSCDetIdAccessor_H
#define DataFormats_CSCDetIdAccessor_H

/** \class CSCChamberIdComparator
 *  Comparator to retrieve CSCrechits by chamber.
 *
 *  $Date: 2006/03/30 $
 *  \author Matteo Sani
 */
// class CSCDetIdComparator {
//public:
// virtual bool operator()( CSCDetId i1, CSCDetId i2 ) const =0;
//};

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
 // typedef std::pair<CSCDetId,CSCDetSameIdComparator&> returnType;
  CSCDetIdAccessor();
  std::pair<CSCDetId,CSCDetIdSameChamberComparator> cscChamber(CSCDetId chamber);
  
private:
  
};

#endif
