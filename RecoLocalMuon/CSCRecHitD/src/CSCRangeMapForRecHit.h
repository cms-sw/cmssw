#ifndef CSCRecHitD_CSCRangeMapForRecHit_H
#define CSCRecHitD_CSCRangeMapForRecHit_H

/** \class CSCRangeMapForRecHit
 *  Comparator to retrieve CSC rechits by chamber or by layer. 
 *
 *  \author Dominique Fortin
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCDetIdSameDetLayerCompare {
 public:
  bool operator() (CSCDetId i1, CSCDetId i2) const;
};

class CSCDetIdSameChamberCompare {
 public:
  bool operator()(CSCDetId i1, CSCDetId i2) const;
};

class CSCRangeMapForRecHit {
 public:
  
  /// Constructor
  CSCRangeMapForRecHit();

  /// Destructor
  virtual ~CSCRangeMapForRecHit();
  
  ///  Returns a valid DetId + a valid comparator for the RangeMap.
  static std::pair<CSCDetId,CSCDetIdSameChamberCompare> cscChamber(CSCDetId id);
  static std::pair<CSCDetId,CSCDetIdSameDetLayerCompare> cscDetLayer(CSCDetId id);

 private:
   
};
 
#endif

