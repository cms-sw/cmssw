#include "FWCore/Framework/interface/CurrentProcessingContext.h"

#include "DataFormats/Common/interface/ModuleDescription.h"

using namespace std;

namespace edm
{
  CurrentProcessingContext::CurrentProcessingContext() :
    pathInSchedule_(0),
    slotInPath_(0),
    moduleDescription_(0),
    pathName_(0)
  { }
  
  string const*
  CurrentProcessingContext::moduleLabel() const
  {
    return moduleDescription_
      ? &(moduleDescription_->moduleLabel_)
      : 0;
  }

  ModuleDescription const*
  CurrentProcessingContext::moduleDescription() const
  {
    return moduleDescription_;
  }

  void
  CurrentProcessingContext::activate(ModuleDescription const* mod,
				     string const*       pathName,
				     size_t              pathInSchedule,
				     size_t              slotInPath)
  {
    assert( mod );
    assert( pathName );
    pathInSchedule_ = pathInSchedule;
    slotInPath_     = slotInPath;
    moduleDescription_ = mod;
    pathName_          = pathName;
  }

  void
  CurrentProcessingContext::deactivate()
  {
    pathInSchedule_    = 0;
    slotInPath_        = 0;
    moduleDescription_ = 0;
    pathName_          = 0;
  }
}
