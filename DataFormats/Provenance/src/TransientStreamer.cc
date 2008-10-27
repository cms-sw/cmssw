#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/EventEntryDescription.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Transient.h"
#include "DataFormats/Provenance/interface/TransientStreamer.h"

namespace edm {
  void setTransientStreamers() {
    SetTransientStreamer<Transient<BranchDescription::Transients> >();
    SetTransientStreamer<Transient<EventEntryDescription::Transients> >();
    SetTransientStreamer<Transient<EventEntryInfo::Transients> >();
    SetTransientStreamer<Transient<FileIndex::Transients> >();
    SetTransientStreamer<Transient<ProcessHistory::Transients> >();
    SetTransientStreamer<Transient<ProductRegistry::Transients> >();
  }
}
