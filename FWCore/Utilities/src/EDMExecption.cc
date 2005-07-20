
#include "FWCore/Utilities/interface/EDMException.h"

#define MAP_ENTRY(name) trans_[edm::errors::name]=#name

namespace edm
{
  template<> void Exception::loadTable()
  {
    MAP_ENTRY(Unknown);
    MAP_ENTRY(ProductNotFound);
    MAP_ENTRY(InsertFailure);
    MAP_ENTRY(Configuration);
    MAP_ENTRY(LogicError);
    MAP_ENTRY(InvalidReference);
    MAP_ENTRY(ModuleFailure);
    MAP_ENTRY(ScheduleExecutionFailure);
    MAP_ENTRY(EventProcessorFailure);
    MAP_ENTRY(NotFound);
  }
 
}
