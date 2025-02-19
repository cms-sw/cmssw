#ifndef DataFormats_Provenance_EventSelectionID_h
#define DataFormats_Provenance_EventSelectionID_h

/*----------------------------------------------------------------------

EventSelectionID: An identifier to uniquely identify the configuration
of the event selector subsystem of an OutputModule.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include <vector>

namespace edm
{
  typedef ParameterSetID EventSelectionID;
  typedef std::vector<EventSelectionID> EventSelectionIDVector;
}

#endif
