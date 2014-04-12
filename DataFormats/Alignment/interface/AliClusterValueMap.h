#ifndef DataFormatsAlignment_AliClusterValueMap_h
#define DataFormatsAlignment_AliClusterValueMap_h

#include "DataFormats/Common/interface/ValueMap.h"
// The content of the container:
#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
// The typedef of the container:
#include "DataFormats/Alignment/interface/AliClusterValueMapFwd.h"

// Still used in AlignmentPrescaler, but to be removed
// (no dictionary generated).
typedef edm::ValueMap<int> AliTrackTakenClusterValueMap;

#endif
