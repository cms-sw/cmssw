#ifndef Geometry_MTDCommonData_MTDTopologyMode_H
#define Geometry_MTDCommonData_MTDTopologyMode_H

#include "FWCore/Utilities/interface/Exception.h"
#include <map>
#include <string>
#include <algorithm>

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

namespace MTDTopologyMode {

  enum class Mode {
    undefined = 0,
    tile = 1,
    bar = 2,
    barzflat = 3,
    barphiflat = 4,
    btlv1etlv4 = 5,
    btlv1etlv5 = 6,
    btlv2etlv5 = 7
  };

  Mode MTDStringToEnumParser(const std::string&);

  /** Returns BTLDetId::CrysLayout as a function of topology mode (to accomodate TDR/post TDR ETL scenarios). **/

  inline BTLDetId::CrysLayout crysLayoutFromTopoMode(const int& topoMode) {
    if (topoMode < 0 || topoMode > static_cast<int>(Mode::btlv2etlv5)) {
      throw cms::Exception("UnknownMTDtopoMode") << "Unknown MTD topology mode " << topoMode;
    } else if (topoMode <= static_cast<int>(BTLDetId::CrysLayout::barphiflat)) {
      return static_cast<BTLDetId::CrysLayout>(topoMode);
    } else if (topoMode < static_cast<int>(Mode::btlv2etlv5)) {
      return BTLDetId::CrysLayout::barphiflat;
    } else {
      return BTLDetId::CrysLayout::tdr;
    }
  }

  /** Returns ETLDetId::EtlLayout as a function of topology mode **/

  inline ETLDetId::EtlLayout etlLayoutFromTopoMode(const int& topoMode) {
    if (topoMode < 0 || topoMode > static_cast<int>(Mode::btlv2etlv5)) {
      throw cms::Exception("UnknownMTDtopoMode") << "Unknown MTD topology mode " << topoMode;
    } else if (topoMode <= static_cast<int>(BTLDetId::CrysLayout::barphiflat)) {
      return ETLDetId::EtlLayout::tp;
    } else if (topoMode == static_cast<int>(Mode::btlv1etlv4)) {
      return ETLDetId::EtlLayout::v4;
    } else {
      return ETLDetId::EtlLayout::v5;
    }
  }

}  // namespace MTDTopologyMode

#endif  // Geometry_MTDCommonData_MTDTopologyMode_H
