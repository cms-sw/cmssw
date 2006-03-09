#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"
#include "CondFormats/OptAlignObjects/interface/XXXXMeasurements.h"
#include "CondFormats/OptAlignObjects/interface/XXXXMeasurementInfo.h"
#include "CondFormats/OptAlignObjects/interface/CSCZSensors.h"

#include "FWCore/EDProduct/interface/Wrapper.h"

#include <string>
#include <vector>

template std::vector<OpticalAlignInfo>::iterator;
template std::vector<XXXXMeasurementInfo>::iterator;
template std::vector<CSCZSensorData>::iterator;
template std::vector<OpticalAlignParam>::iterator;
template std::vector< int >::iterator;
template std::vector< int >::const_iterator;
//template edm::Wrapper<OpticalAlignments>;
namespace {
  namespace {
    edm::Wrapper<OpticalAlignments> tw;
    edm::Wrapper<XXXXMeasurements> tw2;
  }
}
