#ifndef DataFormats_CSCRecHit_CSCRangeMapAccessor_H
#define DataFormats_CSCRecHit_CSCRangeMapAccessor_H

/** \class CSCRangeMapAccessor
 *  Comparator to retrieve CSCrechits by chamber. 
 *
 *  $Date: 2006/07/18 08:36:37 $
 *  \author Matteo Sani
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCDetIdSameDetLayerComparator {
 public:
  bool operator() (CSCDetId i1, CSCDetId i2) const;
};

class CSCDetIdSameChamberComparator {
 public:
  bool operator()(CSCDetId i1, CSCDetId i2) const;
};

class CSCRangeMapAccessor {
 public:
  
  /// Constructor
  CSCRangeMapAccessor();

  /// Destructor
  virtual ~CSCRangeMapAccessor();
  
  ///  Returns a valid DetId + a valid comparator for the RangeMap.
  static std::pair<CSCDetId,CSCDetIdSameChamberComparator> cscChamber(CSCDetId id);
  static std::pair<CSCDetId,CSCDetIdSameDetLayerComparator> cscDetLayer(CSCDetId id);

 private:
   
};
 
#endif
