#ifndef GUARD_HDQMInspectorConfigTracking_h
#define GUARD_HDQMInspectorConfigTracking_h

#include <cstdint>
#include <string>

#include "DQMServices/Diagnostic/interface/HDQMInspectorConfigBase.h"

class HDQMInspectorConfigTracking : public HDQMInspectorConfigBase {
public:
  HDQMInspectorConfigTracking();
  ~HDQMInspectorConfigTracking() override;
};
#endif
