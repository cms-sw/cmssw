#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"

//#include "CondFormats/OptAlignObjects/interface/MBAFork.h"
#include "CondFormats/OptAlignObjects/interface/MBAChBenchCalPlate.h"
#include "CondFormats/OptAlignObjects/interface/MBAChBenchSurveyPlate.h"

#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurements.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurementInfo.h"
#include "CondFormats/OptAlignObjects/interface/CSCZSensors.h"
#include "CondFormats/OptAlignObjects/interface/CSCRSensors.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "CondFormats/OptAlignObjects/interface/Inclinometers.h"
#include "CondFormats/OptAlignObjects/interface/PXsensors.h"

#include <string>
#include <vector>

//template std::vector<OpticalAlignInfo>::iterator;
//template std::vector<MBAForkData>::iterator;
//template std::vector<MBAChBenchCalPlateData>::iterator;
//template std::vector<MBAChBenchSurveyPlateData>::iterator;
//template std::vector<OpticalAlignMeasurementInfo>::iterator;
//template std::vector<CSCZSensorData>::iterator;
//template std::vector<OpticalAlignParam>::iterator;
///template std::vector< int >::iterator;
//template std::vector< int >::const_iterator;
//template std::vector< Inclinometers::Item >::iterator;
//template std::vector< Inclinometers::Item >::const_iterator;
//template std::vector< PXsensors::Item >::iterator;
//template std::vector< PXsensors::Item >::const_iterator;
//template edm::Wrapper<OpticalAlignments>;

namespace{
  struct dictionary {
    std::vector<OpticalAlignInfo> optaligninfovec;
    std::vector<MBAChBenchCalPlateData> mbacalvec;
    std::vector<MBAChBenchSurveyPlateData> mbasurveyvec;
    std::vector<OpticalAlignMeasurementInfo> optmeasureinfovec;
    std::vector<CSCZSensorData> zsensorvec;
    std::vector<CSCRSensorData> rsensorvec;
    std::vector<OpticalAlignParam> OpticalAlignParamvec;
    std::vector< Inclinometers::Item > incvec;
    std::vector< PXsensors::Item > pxvec;

    edm::Wrapper<OpticalAlignments> tw;
    edm::Wrapper<OpticalAlignMeasurements> tw2;
  };
}
