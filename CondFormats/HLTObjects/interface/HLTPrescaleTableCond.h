#ifndef CondFormats_HLTObjects_HLTPrescaleTableCond_h
#define CondFormats_HLTObjects_HLTPrescaleTableCond_h

#include "DataFormats/HLTReco/interface/HLTPrescaleTable.h"

namespace trigger {
  class HLTPrescaleTableCond {
   public:
    /// default c'tor
    HLTPrescaleTableCond(): hltPrescaleTable_() { }
    /// payload c'tor
    HLTPrescaleTableCond(const trigger::HLTPrescaleTable& hltPrescaleTable):
	hltPrescaleTable_(hltPrescaleTable) { }
    /// trivial const accessor
    const trigger::HLTPrescaleTable& hltPrescaleTable() const {return hltPrescaleTable_;}
    /// data member
    trigger::HLTPrescaleTable hltPrescaleTable_;
  };
}
#endif
