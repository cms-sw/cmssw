#ifndef GUARD_HDQMInspectorConfigSiPixel_h
#define GUARD_HDQMInspectorConfigSiPixel_h

#include <string>

#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelSummary.h"
#include "DQMServices/Diagnostic/interface/HDQMInspectorConfigBase.h"


class HDQMInspectorConfigSiPixel : public HDQMInspectorConfigBase
{
 public:
  HDQMInspectorConfigSiPixel ();
  virtual ~HDQMInspectorConfigSiPixel ();
    
  std::string translateDetId (const uint32_t) const;
};



#endif
