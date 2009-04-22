#include <sstream>

#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

#include "CondFormats/SiPixelObjects/interface/PixelCaenAliasEndcap.h"

PixelCaenAliasEndcap::PixelCaenAliasEndcap(const DetId& id)
{
  std::ostringstream name;

  name << "CMS_Pixel/Endcap/B";

  PixelEndcapName endcap(id);

  switch ( endcap.halfCylinder() )
  {
    case PixelEndcapName::mO: { name << "mO"; break; }
    case PixelEndcapName::mI: { name << "mI"; break; }
    case PixelEndcapName::pO: { name << "pO"; break; }
    case PixelEndcapName::pI: { name << "pI"; }
  }

  name << "/D" << endcap.diskName()
       << "/ROG" << (endcap.bladeName() - 1) / 3 + 1;

  std::string parent = name.str();

  theDigitalLV = parent + "/LV/Digital";
  theAnalogLV  = parent + "/LV/Analog";
  theBiasedHV  = parent + "/HV/Bias_";
  theBiasedHV += endcap.plaquetteName() < 3 ? "innerRadius" : "outerRadius";
}
