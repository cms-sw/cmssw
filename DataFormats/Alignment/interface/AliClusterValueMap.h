#ifndef CommonAlignment_AliHitAssoMapV_h
#define CommonAlignment_AliHitAssoMapV_h

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include <vector>
#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"



typedef edm::ValueMap<AlignmentClusterFlag >   AliClusterValueMap; 
typedef edm::ValueMap<int >   AliTrackTakenClusterValueMap; 

#endif
