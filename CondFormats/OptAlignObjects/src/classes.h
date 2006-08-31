#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"

#include "CondFormats/OptAlignObjects/interface/MBAChBenchCalPlate.h"
#include "CondFormats/OptAlignObjects/interface/MBAChBenchSurveyPlate.h"

#include "CondFormats/OptAlignObjects/interface/CSCZSensors.h"
#include "CondFormats/OptAlignObjects/interface/CSCRSensors.h"

#include "DataFormats/Common/interface/Wrapper.h"

#include <string>
#include <vector>

namespace {
  namespace {
    edm::Wrapper<OpticalAlignments> tw;
    std::vector<OpticalAlignInfo> opinfo;
    std::vector<MBAChBenchCalPlateData> plate;
    std::vector<CSCZSensorData> z;
    std::vector<CSCRSensorData> r;
    std::vector<OpticalAlignParam> pa;
    std::vector< int > i;
  }
}

