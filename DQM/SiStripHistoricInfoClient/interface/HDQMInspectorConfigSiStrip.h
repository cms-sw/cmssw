#ifndef GUARD_HDQMInspectorConfigSiStrip_h
#define GUARD_HDQMInspectorConfigSiStrip_h

#include <string>
#include <stdint.h>

#include "DQMServices/Diagnostic/interface/HDQMInspectorConfigBase.h"


class HDQMInspectorConfigSiStrip : public HDQMInspectorConfigBase
{
  public:
    HDQMInspectorConfigSiStrip ();
    ~HDQMInspectorConfigSiStrip () override;
    
    std::string translateDetId (const uint32_t) const override;

};



#endif
