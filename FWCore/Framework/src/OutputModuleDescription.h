#ifndef FWCore_Framework_OutputModuleDescription_h
#define FWCore_Framework_OutputModuleDescription_h

/*----------------------------------------------------------------------

OutputModuleDescription : the stuff that is needed to configure an
output module that does not come in through the ParameterSet  

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchIDList.h"
namespace edm {

  class BranchIDListHelper;
  class SubProcessParentageHelper;

  struct OutputModuleDescription {
    //OutputModuleDescription() : maxEvents_(-1) {}
    explicit OutputModuleDescription(BranchIDLists const& branchIDLists,
                                     int maxEvents = -1,
                                     SubProcessParentageHelper const* subProcessParentageHelper = nullptr)
        : branchIDLists_(&branchIDLists),
          maxEvents_(maxEvents),
          subProcessParentageHelper_(subProcessParentageHelper) {}
    BranchIDLists const* branchIDLists_;
    int maxEvents_;
    SubProcessParentageHelper const* subProcessParentageHelper_;
  };
}  // namespace edm

#endif
