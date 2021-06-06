#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/GeometryObjects/src/headers.h"

int main() {
  testSerialization<CSCRecoDigiParameters>();
  testSerialization<PCaloGeometry>();
  testSerialization<PGeometricDet>();
  //testSerialization<PGeometricDet::Item>(); has uninitialized booleans
  testSerialization<RecoIdealGeometry>();
  testSerialization<std::vector<PGeometricDet::Item>>();
  testSerialization<PTrackerParameters>();
  testSerialization<PTrackerParameters::Item>();
  testSerialization<HcalParameters>();
  testSerialization<PHGCalParameters>();
  testSerialization<PMTDParameters>();
  testSerialization<HcalSimulationParameters>();
  testSerialization<PDetGeomDesc>();

  return 0;
}
