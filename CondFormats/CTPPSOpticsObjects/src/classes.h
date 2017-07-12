#include "CondFormats/CTPPSOpticsObjects/interface/LHCOpticsApproximator.h"
#include "CondFormats/CTPPSOpticsObjects/interface/TMultiDimFet.h"

namespace CondFormats_CTPPSOpticsObjects
{
  struct dictionary
  {
    LHCOpticsApproximator loa;
    LHCApertureApproximator laa;
    TMultiDimFet mdf;
  };
}
