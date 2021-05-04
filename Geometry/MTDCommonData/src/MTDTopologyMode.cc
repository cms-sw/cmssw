#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include <string>

namespace MTDTopologyMode {

  Mode MTDStringToEnumParser(const std::string &value) {
    std::string prefix("MTDTopologyMode::");
    Mode output = Mode::undefined;
    if (value == prefix + "tile") {
      output = Mode::tile;
    } else if (value == prefix + "bar") {
      output = Mode::bar;
    } else if (value == prefix + "barzflat") {
      output = Mode::barzflat;
    } else if (value == prefix + "barphiflat") {
      output = Mode::barphiflat;
    } else if (value == prefix + "btlv1etlv4") {
      output = Mode::btlv1etlv4;
    } else if (value == prefix + "btlv1etlv5") {
      output = Mode::btlv1etlv5;
    } else {
      throw cms::Exception("MTDTopologyModeError") << "the value " << value << " is not defined.";
    }
    return output;
  }

}  // namespace MTDTopologyMode
