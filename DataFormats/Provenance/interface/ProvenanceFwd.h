#ifndef DataFormats_Provenance_ProvenanceFwd_h
#define DataFormats_Provenance_ProvenanceFwd_h

namespace edm {
  class BranchDescription;
  class BranchIDListHelper;
  class Parentage;
  class BranchID;
  class BranchKey;
  class BranchDescription;
  class ProductProvenance;
  class ProductProvenance;
  class EventAuxiliary;
  class EventID;
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
  class Timestamp;
  class ProductProvenanceRetriever;
}  // namespace edm

namespace cms {
  class Exception;  // In FWCore/Utilities
}

#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/PassID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#endif
