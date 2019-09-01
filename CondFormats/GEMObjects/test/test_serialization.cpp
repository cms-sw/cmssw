#include "CondFormats/Serialization/interface/Test.h"
#include "CondFormats/GEMObjects/src/headers.h"

int main() {
  testSerialization<GEMELMap>();
  testSerialization<GEMELMap::GEMVFatMap>();
  testSerialization<std::vector<GEMELMap::GEMVFatMap>>();
  testSerialization<GEMELMap::GEMStripMap>();
  testSerialization<std::vector<GEMELMap::GEMStripMap>>();

  testSerialization<GEMeMap>();
  testSerialization<GEMeMap::GEMChamberMap>();
  testSerialization<std::vector<GEMeMap::GEMChamberMap>>();
  testSerialization<GEMeMap::GEMVFatMap>();
  testSerialization<std::vector<GEMeMap::GEMVFatMap>>();
  testSerialization<GEMeMap::GEMStripMap>();
  testSerialization<std::vector<GEMeMap::GEMStripMap>>();

  testSerialization<GEMDeadStrips>();
  testSerialization<GEMDeadStrips::DeadItem>();
  testSerialization<GEMMaskedStrips>();
  testSerialization<GEMMaskedStrips::MaskItem>();
}
