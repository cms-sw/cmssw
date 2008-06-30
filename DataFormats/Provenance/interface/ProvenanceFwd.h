#ifndef DataFormats_Provenance_ProvenanceFwd_h
#define DataFormats_Provenance_ProvenanceFwd_h

namespace edm {
  class BranchDescription;
  class EventEntryDescription;
  class BranchID;
  class BranchKey;
  class ConstBranchDescription;
  class EventEntryInfo;
  class RunLumiEntryInfo;
  class BranchKey;
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
  class Timestamp;
  template <typename T> class BranchMapper;
}
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"
#include "DataFormats/Provenance/interface/PassID.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationRegistry.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionID.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#endif
