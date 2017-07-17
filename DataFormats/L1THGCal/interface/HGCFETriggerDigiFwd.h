#ifndef __DataFormats_L1THGCal_HGCFETriggerDigiFwd_h__
#define __DataFormats_L1THGCal_HGCFETriggerDigiFwd_h__

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/Common/interface/SortedCollection.h"

namespace l1t {
  //fwd decl. of FETriggerDigi
  class HGCFETriggerDigi;

  // main collection type
  typedef edm::SortedCollection<HGCFETriggerDigi> HGCFETriggerDigiCollection;

  // refs and ptrs
  typedef edm::Ref<HGCFETriggerDigiCollection> HGCFETriggerDigiRef;
  typedef edm::RefVector<HGCFETriggerDigiCollection> HGCFETriggerDigiRefVector;
  typedef edm::Ptr<HGCFETriggerDigi> HGCFETriggerDigiPtr;
  typedef std::vector<HGCFETriggerDigiPtr> HGCFETriggerDigiPtrVector;
}

#endif
