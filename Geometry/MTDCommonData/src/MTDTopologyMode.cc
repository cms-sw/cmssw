#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include <string>

namespace MTDTopologyMode {

  Mode MTDStringToEnumParser(const std::string &value) {
    std::string prefix("MTDTopologyMode::");
    Mode output = Mode::undefined;
    if (value == prefix + "btlv2etlv5") {
      output = Mode::btlv2etlv5;
    } else if (value == prefix + "btlv3etlv8") {
      output = Mode::btlv3etlv8;
    } else if (value == prefix + "btlv4etlv8") {
      output = Mode::btlv4etlv8;
    } else if (value == prefix + "btlv4etlv9") {
      output = Mode::btlv4etlv9;
    } else if (value == prefix + "btlv4etlv10") {
      output = Mode::btlv4etlv10;
    } else {
      throw cms::Exception("MTDTopologyModeError") << "the value " << value << " is not defined.";
    }
    return output;
  }

}  // namespace MTDTopologyMode
