#ifndef DQM_HGCAL_interface_HGCalDQMWorkerBase_h
#define DQM_HGCAL_interface_HGCalDQMWorkerBase_h

#include "DQMServices/Core/interface/DQMStore.h"

namespace edm {
  class LuminosityBlock;
}

namespace hgcal {
  namespace dqm {

    // Matches DQMEDHarvester's own alias so worker signatures are identical.
    using DQMStore = ::dqm::harvesting::DQMStore;

    class HGCalDQMGeometry;

    // endLumi/endRun take non-const geometry: fills may open template files
    // via moduleTemplateFile / trigTemplateFile, which mutate the TFile cache.
    class HGCalDQMWorkerBase {
    public:
      virtual ~HGCalDQMWorkerBase() = default;
      virtual void book(DQMStore::IBooker&, HGCalDQMGeometry const&) = 0;
      virtual void endLumi(DQMStore::IBooker&, DQMStore::IGetter&, HGCalDQMGeometry&, edm::LuminosityBlock const&) {}
      virtual void endRun(DQMStore::IBooker&, DQMStore::IGetter&, HGCalDQMGeometry&) {}
    };

  }  // namespace dqm
}  // namespace hgcal

#endif
