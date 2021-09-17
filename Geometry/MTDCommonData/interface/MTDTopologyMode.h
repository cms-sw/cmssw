#ifndef Geometry_MTDCommonData_MTDTopologyMode_H
#define Geometry_MTDCommonData_MTDTopologyMode_H

#include "FWCore/Utilities/interface/Exception.h"
#include <map>
#include <string>
#include <algorithm>

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

namespace MTDTopologyMode {

  enum class Mode { undefined = 0, tile = 1, bar = 2, barzflat = 3, barphiflat = 4, btlv1etlv4 = 5, btlv1etlv5 = 6 };

  Mode MTDStringToEnumParser(const std::string&);

  /** Returns BTLDetId::CrysLayout as a function of topology mode (to accomodate TDR/post TDR ETL scenarios). **/

  inline BTLDetId::CrysLayout crysLayoutFromTopoMode(const int& topoMode) {
    return (topoMode <= static_cast<int>(BTLDetId::CrysLayout::barphiflat) ? static_cast<BTLDetId::CrysLayout>(topoMode)
                                                                           : BTLDetId::CrysLayout::barphiflat);
  }

}  // namespace MTDTopologyMode

#endif  // Geometry_MTDCommonData_MTDTopologyMode_H
