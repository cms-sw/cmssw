#ifndef FWCore_Framework_InputSourceDescription_h
#define FWCore_Framework_InputSourceDescription_h

/*----------------------------------------------------------------------

InputSourceDescription : the stuff that is needed to configure an
input source that does not come in through the ParameterSet  
----------------------------------------------------------------------*/
#include "boost/shared_ptr.hpp"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

namespace edm {
  class ProductRegistry;
  class ActivityRegistry;
  class BranchIDListHelper;

  struct InputSourceDescription {
    InputSourceDescription() :
      moduleDescription_(),
      productRegistry_(nullptr),
      actReg_(),
      maxEvents_(-1),
      maxLumis_(-1),
      nStreams_(1U) {
    }

    InputSourceDescription(ModuleDescription const& md,
                           ProductRegistry& preg,
                           boost::shared_ptr<BranchIDListHelper> branchIDListHelper,
                           boost::shared_ptr<ActivityRegistry> areg,
                           int maxEvents,
                           int maxLumis,
                           unsigned int nStreams = 1U) :
      moduleDescription_(md),
      productRegistry_(&preg),
      branchIDListHelper_(branchIDListHelper),
      actReg_(areg),
      maxEvents_(maxEvents),
      maxLumis_(maxLumis),
      nStreams_(nStreams) {
   }

    ModuleDescription moduleDescription_;
    ProductRegistry* productRegistry_;
    boost::shared_ptr<BranchIDListHelper> branchIDListHelper_;
    boost::shared_ptr<ActivityRegistry> actReg_;
    int maxEvents_;
    int maxLumis_;
    unsigned int nStreams_;
  };
}

#endif
