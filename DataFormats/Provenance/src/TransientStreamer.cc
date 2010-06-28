#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Transient.h"
#include "DataFormats/Provenance/interface/TransientStreamer.h"

namespace edm {
  void setTransientStreamers() {
    SetTransientStreamer<Transient<BranchDescription::Transients> >();
    SetTransientStreamer<Transient<ProductProvenance::Transients> >();
    SetTransientStreamer<Transient<IndexIntoFile::Transients> >();
    SetTransientStreamer<Transient<Parentage::Transients> >();
    SetTransientStreamer<Transient<ProcessConfiguration::Transients> >();
    SetTransientStreamer<Transient<ProcessHistory::Transients> >();
    SetTransientStreamer<Transient<ProductRegistry::Transients> >();
  }
}
