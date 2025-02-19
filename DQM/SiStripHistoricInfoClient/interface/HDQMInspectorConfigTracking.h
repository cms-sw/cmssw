#ifndef GUARD_HDQMInspectorConfigTracking_h
#define GUARD_HDQMInspectorConfigTracking_h

#include <string>
#include <stdint.h>

#include "DQMServices/Diagnostic/interface/HDQMInspectorConfigBase.h"

class HDQMInspectorConfigTracking : public HDQMInspectorConfigBase
{
  public:
    HDQMInspectorConfigTracking ();
    virtual ~HDQMInspectorConfigTracking ();

    std::string translateDetId (const uint32_t) const;
};



#endif
