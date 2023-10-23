#ifndef DataFormats_LaserAlignment_AlignmentClusterFlag_h
#define DataFormats_LaserAlignment_AlignmentClusterFlag_h

#include "DataFormats/DetId/interface/DetId.h"

/**
 * Class that defines a flag for each cluster used by the alignment
 * The flag contains informations used to categorise and (eventually)
 * decide whether to use the hit for the final alignment. This informations
 * are bit-packed into a 8-bit word.
 *
 * Original author: A. Bonato
 */

class AlignmentClusterFlag {
public:
  AlignmentClusterFlag();
  AlignmentClusterFlag(const DetId &id);

  bool isTaken() const;
  bool isOverlap() const;
  void SetTakenFlag();
  void SetOverlapFlag();
  void SetDetId(const DetId &newdetid);
  const DetId &detId() const { return detId_; }
  char hitFlag() const { return hitFlag_; }

private:
  DetId detId_;
  char hitFlag_;
};
#endif
