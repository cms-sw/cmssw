#ifndef __L1TMuonEndCap_TriggerPrimitiveFwd_h__
#define __L1TMuonEndCap_TriggerPrimitiveFwd_h__

#include <vector>
#include <map>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

namespace L1TMuonEndCap {
  class TriggerPrimitive;

  typedef std::vector<TriggerPrimitive> TriggerPrimitiveCollection;

  //typedef edm::Ref<TriggerPrimitiveCollection> TriggerPrimitiveRef;
  //typedef std::vector<TriggerPrimitiveRef> TriggerPrimitiveList;
  //typedef edm::Ptr<TriggerPrimitive> TriggerPrimitivePtr;
  //typedef std::map<unsigned,TriggerPrimitiveCollection> TriggerPrimitiveStationMap;

  class TTTriggerPrimitive;  // Track Trigger hits

  typedef std::vector<TTTriggerPrimitive> TTTriggerPrimitiveCollection;
}

#endif
