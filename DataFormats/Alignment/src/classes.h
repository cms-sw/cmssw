#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"
#include <vector>
#include <map>

namespace {
  namespace {
    AlignmentClusterFlag                                 ahf;
     edm::Wrapper<AlignmentClusterFlag>                   wahf;
    AliClusterValueMap                                        ahvm1;
     edm::Wrapper<AliClusterValueMap>                          wahvm1;
    //  AliTrackTakenClusterValueMap                              atthvm1;
    //  edm::Wrapper<AliTrackTakenClusterValueMap>                watthvm1;
  }
}
