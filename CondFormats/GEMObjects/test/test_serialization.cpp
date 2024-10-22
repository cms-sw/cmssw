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

  testSerialization<GEMChMap>();
  testSerialization<GEMChMap::sectorEC>();
  testSerialization<GEMChMap::chamEC>();
  testSerialization<GEMChMap::chamDC>();
  testSerialization<GEMChMap::vfatEC>();
  testSerialization<GEMChMap::channelNum>();
  testSerialization<GEMChMap::stripNum>();
  testSerialization<std::vector<GEMChMap::sectorEC>>();
  testSerialization<std::map<GEMChMap::chamEC, GEMChMap::chamDC>>();
  testSerialization<std::map<int, std::vector<uint16_t>>>();
  testSerialization<std::map<GEMChMap::vfatEC, std::vector<int>>>();
  testSerialization<std::map<GEMChMap::channelNum, GEMChMap::stripNum>>();
  testSerialization<std::map<GEMChMap::stripNum, GEMChMap::channelNum>>();

  testSerialization<GEMDeadStrips>();
  testSerialization<GEMDeadStrips::DeadItem>();
  testSerialization<GEMMaskedStrips>();
  testSerialization<GEMMaskedStrips::MaskItem>();
}
