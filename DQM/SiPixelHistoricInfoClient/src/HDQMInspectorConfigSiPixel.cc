#include "DQM/SiPixelHistoricInfoClient/interface/HDQMInspectorConfigSiPixel.h"

HDQMInspectorConfigSiPixel::HDQMInspectorConfigSiPixel ()
{
}

HDQMInspectorConfigSiPixel::~HDQMInspectorConfigSiPixel ()
{
}

std::string HDQMInspectorConfigSiPixel::translateDetId(const uint32_t id) const
{
  switch (id) {
    case sipixelsummary::TRACKER:
      return "TRACKER";
    case sipixelsummary::Barrel:
      return "Barrel";
    case sipixelsummary::Shell_mI:
      return "Shell_mI";
    case sipixelsummary::Shell_mO:
      return "Shell_mO";
    case sipixelsummary::Shell_pI:
      return "Shell_pI";
    case sipixelsummary::Shell_pO:
      return "Shell_pO";
    case sipixelsummary::Endcap:
      return "Endcap";
    case sipixelsummary::HalfCylinder_mI:
      return "HalfCylinder_mI";
    case sipixelsummary::HalfCylinder_mO:
      return "HalfCylinder_mO";
    case sipixelsummary::HalfCylinder_pI:
      return "HalfCylinder_pI";
    case sipixelsummary::HalfCylinder_pO:
      return "HalfCylinder_pO";
    default:
      return "???";
  };
}
