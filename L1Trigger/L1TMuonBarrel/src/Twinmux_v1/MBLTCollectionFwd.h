//Authors:
// Carlo Battilana - Giuseppe Codispoti

#ifndef __L1ITMU_MBLTCollection_H__
#define __L1ITMU_MBLTCollection_H__

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

class DTChamberId;

namespace L1TwinMux {
  class MBLTCollection;
  typedef std::pair<DTChamberId, MBLTCollection> MBLTContainerPair;
  typedef std::map<DTChamberId, MBLTCollection> MBLTContainer;
  typedef edm::Ref<MBLTContainer> MBLTContainerRef;
//   typedef std::vector<MBLTContainerRef> MBLTVectorRef;
  typedef std::pair<MBLTContainerRef,TriggerPrimitiveRef> MBLTContainerRefPair;
  typedef std::vector<MBLTContainerRefPair> MBLTVectorRef;

}

#endif
