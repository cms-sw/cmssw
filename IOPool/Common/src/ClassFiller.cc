#include "IOPool/Common/interface/ClassFiller.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/EventEntryDescription.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/RootAutoLibraryLoader/interface/TransientStreamer.h"
#include "Cintex/Cintex.h"
#include "TH1.h"
#include "G__ci.h"
#include "TTree.h"

namespace edm {
  // ---------------------
  void ClassFiller() {
    TTree::SetMaxTreeSize(kMaxLong64);
    TH1::AddDirectory(kFALSE);
    G__SetCatchException(0);
    ROOT::Cintex::Cintex::Enable();
    SetTransientStreamer<Transient<BranchDescription::Transients> >();
    SetTransientStreamer<Transient<EventEntryDescription::Transients> >();
    SetTransientStreamer<Transient<EventEntryInfo::Transients> >();
    SetTransientStreamer<Transient<FileIndex::Transients> >();
    SetTransientStreamer<Transient<ProcessHistory::Transients> >();
    SetTransientStreamer<Transient<ProductRegistry::Transients> >();
  }
}
