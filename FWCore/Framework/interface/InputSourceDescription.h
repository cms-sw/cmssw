#ifndef FWCore_Framework_InputSourceDescription_h
#define FWCore_Framework_InputSourceDescription_h

/*----------------------------------------------------------------------

InputSourceDescription : the stuff that is needed to configure an
input source that does not come in through the ParameterSet  
----------------------------------------------------------------------*/
#include <memory>
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"

namespace edm {
  class ProductRegistry;
  class ActivityRegistry;
  class BranchIDListHelper;
  class PreallocationConfiguration;
  class ThinnedAssociationsHelper;

  struct InputSourceDescription {
    InputSourceDescription()
        : moduleDescription_(),
          productRegistry_(nullptr),
          actReg_(),
          maxEvents_(-1),
          maxLumis_(-1),
          allocations_(nullptr) {}

    InputSourceDescription(ModuleDescription const& md,
                           std::shared_ptr<ProductRegistry> preg,
                           std::shared_ptr<BranchIDListHelper> branchIDListHelper,
                           std::shared_ptr<ProcessBlockHelper> const& processBlockHelper,
                           std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper,
                           std::shared_ptr<ActivityRegistry> areg,
                           int maxEvents,
                           int maxLumis,
                           int maxSecondsUntilRampdown,
                           PreallocationConfiguration const& allocations)
        : moduleDescription_(md),
          productRegistry_(preg),
          branchIDListHelper_(branchIDListHelper),
          processBlockHelper_(processBlockHelper),
          thinnedAssociationsHelper_(thinnedAssociationsHelper),
          actReg_(areg),
          maxEvents_(maxEvents),
          maxLumis_(maxLumis),
          maxSecondsUntilRampdown_(maxSecondsUntilRampdown),
          allocations_(&allocations) {}

    ModuleDescription moduleDescription_;
    std::shared_ptr<ProductRegistry> productRegistry_;
    std::shared_ptr<BranchIDListHelper> branchIDListHelper_;
    std::shared_ptr<ProcessBlockHelper> processBlockHelper_;
    std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper_;
    std::shared_ptr<ActivityRegistry> actReg_;  // We do not use propagate_const because the registry itself is mutable.
    int maxEvents_;
    int maxLumis_;
    int maxSecondsUntilRampdown_;
    PreallocationConfiguration const* allocations_;
  };
}  // namespace edm

#endif
