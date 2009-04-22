#include <sstream>

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"

#include "CondFormats/SiPixelObjects/interface/PixelCaenAliasBarrel.h"

PixelCaenAliasBarrel::PixelCaenAliasBarrel(const DetId& id)
{
  std::ostringstream name;

  name << "CMS_Pixel/Barrel/B";

  PixelBarrelName barrel(id);

  int shell = barrel.shell();
  int layer = barrel.layerName();

  switch (shell)
  {
    case PixelBarrelName::mO: name << "mO"; break;
    case PixelBarrelName::mI: name << "mI"; break;
    case PixelBarrelName::pO: name << "pO"; break;
    case PixelBarrelName::pI: name << "pI";
  }

  name << "/SEC" << barrel.sectorName()
       << (layer == 3 ? "/LAY3" : "/LAY1and2");

  std::string parent = name.str(); // eg: CMS_Pixel/Barrel/BmO/SEC1/LAY3

  theDigitalLV = parent + "/LV/Digital";
  theAnalogLV  = parent + "/LV/Analog";
  theBiasedHV  = parent;

  if (1 == layer) theBiasedHV += "/HV/Bias_Channel1";
  else
    if (2 == layer) theBiasedHV += "/HV/Bias_Channel2";
  else // Layer 3
    if (PixelBarrelName::mO == shell || PixelBarrelName::pI == shell)
      switch ( barrel.ladderName() )
      {
        case 3:
        case 6:
        case 9:
        case 11:
        case 13:
        case 15:
        case 16:
        case 18:
        case 19:
        case 21:
        case 22:
          theBiasedHV += "/HV/Bias_Channel1";
          break;
        default:
          theBiasedHV += "/HV/Bias_Channel2";
      }
    else // BmI or BpO
      switch ( barrel.ladderName() )
      {
        case 3:
        case 6:
        case 9:
        case 11:
        case 13:
        case 15:
        case 16:
        case 18:
        case 19:
        case 21:
        case 22:
          theBiasedHV += "/HV/Bias_Channel2";
          break;
        default:
          theBiasedHV += "/HV/Bias_Channel1";
      }
}
