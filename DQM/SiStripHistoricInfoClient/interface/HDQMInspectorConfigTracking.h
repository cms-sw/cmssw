#ifndef GUARD_HDQMInspectorConfigTracking_h
#define GUARD_HDQMInspectorConfigTracking_h

#include <string>
#include <cstdint>

#include "DQMServices/Diagnostic/interface/HDQMInspectorConfigBase.h"

class HDQMInspectorConfigTracking : public HDQMInspectorConfigBase
{
  public:
    HDQMInspectorConfigTracking ();
    ~HDQMInspectorConfigTracking () override;
};
#endif
