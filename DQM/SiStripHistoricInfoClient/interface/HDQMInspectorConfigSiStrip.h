#ifndef GUARD_HDQMInspectorConfigSiStrip_h
#define GUARD_HDQMInspectorConfigSiStrip_h

#include <cstdint>
#include <string>

#include "DQMServices/Diagnostic/interface/HDQMInspectorConfigBase.h"

class HDQMInspectorConfigSiStrip : public HDQMInspectorConfigBase {
public:
  HDQMInspectorConfigSiStrip();
  ~HDQMInspectorConfigSiStrip() override;
};
#endif
