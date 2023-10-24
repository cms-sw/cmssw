#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"

AlignmentClusterFlag::AlignmentClusterFlag() : detId_(0), hitFlag_(0) {}

AlignmentClusterFlag::AlignmentClusterFlag(const DetId &id) : detId_(id), hitFlag_(0) {}

bool AlignmentClusterFlag::isTaken() const { return ((hitFlag_ & (1 << 0)) != 0); }

bool AlignmentClusterFlag::isOverlap() const { return ((hitFlag_ & (1 << 1)) != 0); }

void AlignmentClusterFlag::SetTakenFlag() { hitFlag_ |= (1 << 0); }

void AlignmentClusterFlag::SetOverlapFlag() { hitFlag_ |= (1 << 1); }

void AlignmentClusterFlag::SetDetId(const DetId &newdetid) { detId_ = newdetid; }
