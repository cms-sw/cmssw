#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/OptAlignObjects/src/headers.h"

int main() {
  testSerialization<CSCRSensorData>();
  testSerialization<CSCRSensors>();
  testSerialization<CSCZSensorData>();
  testSerialization<CSCZSensors>();
  testSerialization<Inclinometers>();
  testSerialization<Inclinometers::Item>();
  testSerialization<MBAChBenchCalPlate>();
  testSerialization<MBAChBenchCalPlateData>();
  testSerialization<MBAChBenchSurveyPlate>();
  testSerialization<MBAChBenchSurveyPlateData>();
  testSerialization<OpticalAlignInfo>();
  testSerialization<OpticalAlignMeasurementInfo>();
  testSerialization<OpticalAlignMeasurements>();
  testSerialization<OpticalAlignParam>();
  testSerialization<OpticalAlignments>();
  testSerialization<PXsensors>();
  testSerialization<PXsensors::Item>();
  //testSerialization<edm::Wrapper<OpticalAlignMeasurements>>(); no copy-constructor, unused?
  //testSerialization<edm::Wrapper<OpticalAlignments>>(); no copy-constructor, unused?
  testSerialization<std::vector<CSCRSensorData>>();
  testSerialization<std::vector<CSCZSensorData>>();
  testSerialization<std::vector<Inclinometers::Item>>();
  testSerialization<std::vector<MBAChBenchCalPlateData>>();
  testSerialization<std::vector<MBAChBenchSurveyPlateData>>();
  testSerialization<std::vector<OpticalAlignInfo>>();
  testSerialization<std::vector<OpticalAlignMeasurementInfo>>();
  testSerialization<std::vector<OpticalAlignParam>>();
  testSerialization<std::vector<PXsensors::Item>>();

  return 0;
}
