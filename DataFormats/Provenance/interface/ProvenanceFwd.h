#ifndef DataFormats_Provenance_ProvenanceFwd_h
#define DataFormats_Provenance_ProvenanceFwd_h

namespace edm {
  class BranchIDListHelper;
  class Parentage;
  class BranchID;
  class BranchKey;
  class ProductProvenance;
  class EventAuxiliary;
  class EventID;
  class EventToProcessBlockIndexes;
  class LuminosityBlockAuxiliary;
  class LuminosityBlockID;
  class ModuleDescription;
  class ProcessConfiguration;
  class ProcessHistory;
  class ProcessRegistry;
  class ProductID;
  class ProductRegistry;
  class Provenance;
  class RunAuxiliary;
  class RunID;
  class StableProvenance;
  class StoredProcessBlockHelper;
  class Timestamp;
  class ProductProvenanceLookup;
}  // namespace edm

namespace cms {
  class Exception;  // In FWCore/Utilities
}

#include "DataFormats/Provenance/interface/ProductDescriptionFwd.h"
#endif
