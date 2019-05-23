#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"
#include <sstream>

std::string ChamberStripSpec::print(int depth) const {
  std::ostringstream str;
  if (depth >= 0) {
    str << " ChamberStripSpec: "
        << " pin: " << cablePinNumber << ", chamber: " << chamberStripNumber << ", CMS strip: " << cmsStripNumber
        << std::endl;
  }
  return str.str();
}
