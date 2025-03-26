#ifndef Geometry_MTDCommonData_MTDTopologyMode_H
#define Geometry_MTDCommonData_MTDTopologyMode_H

#include "FWCore/Utilities/interface/Exception.h"
#include <map>
#include <string>
#include <algorithm>

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

namespace MTDTopologyMode {

  enum class Mode {
    undefined = 0,
    tile = 1,
    bar = 2,
    barzflat = 3,
    barphiflat = 4,
    btlv1etlv4 = 5,
    btlv1etlv5 = 6,
    btlv2etlv5 = 7,
    btlv3etlv8 = 8,
    btlv4etlv8 = 9,
    btlv4etlv9 = 10,
    btlv4etlv10 = 11
  };

  enum class EtlLayout { v5 = 3, v8 = 4, v9 = 5, v10 = 6 };

  Mode MTDStringToEnumParser(const std::string&);

  /** Returns BTLDetId::CrysLayout as a function of topology mode (to accomodate TDR/post TDR ETL scenarios). **/

  inline BTLDetId::CrysLayout crysLayoutFromTopoMode(const int& topoMode) {
    switch (topoMode) {
      case static_cast<int>(Mode::btlv4etlv10):
        return BTLDetId::CrysLayout::v4;
        break;
      case static_cast<int>(Mode::btlv4etlv9):
        return BTLDetId::CrysLayout::v4;
        break;
      case static_cast<int>(Mode::btlv4etlv8):
        return BTLDetId::CrysLayout::v4;
        break;
      case static_cast<int>(Mode::btlv3etlv8):
        return BTLDetId::CrysLayout::v3;
        break;
      case static_cast<int>(Mode::btlv2etlv5):
        return BTLDetId::CrysLayout::v2;
        break;
      default:
        throw cms::Exception("UnknownMTDtopoMode") << "Unknown MTD topology mode :( " << topoMode;
        break;
    }
  }

  /** Returns EtlLayout as a function of topology mode **/

  inline EtlLayout etlLayoutFromTopoMode(const int& topoMode) {
    switch (topoMode) {
      case static_cast<int>(Mode::btlv4etlv10):
        return EtlLayout::v10;
        break;
      case static_cast<int>(Mode::btlv4etlv9):
        return EtlLayout::v9;
        break;
      case static_cast<int>(Mode::btlv4etlv8):
        return EtlLayout::v8;
        break;
      case static_cast<int>(Mode::btlv3etlv8):
        return EtlLayout::v8;
        break;
      case static_cast<int>(Mode::btlv2etlv5):
        return EtlLayout::v5;
        break;
      default:
        throw cms::Exception("UnknownMTDtopoMode") << "Unknown MTD topology mode :( " << topoMode;
        break;
    }
  }

}  // namespace MTDTopologyMode

#endif  // Geometry_MTDCommonData_MTDTopologyMode_H
