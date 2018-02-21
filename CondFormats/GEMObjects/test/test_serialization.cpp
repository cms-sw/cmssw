#include "CondFormats/Serialization/interface/Test.h"
#include "CondFormats/GEMObjects/src/headers.h"

int main()
{
  testSerialization<GEMEMap>();
  testSerialization<GEMEMap::GEMVFatMap>();
  testSerialization<std::vector<GEMEMap::GEMVFatMap>>();
  testSerialization<GEMEMap::GEMStripMap>();
  testSerialization<std::vector<GEMEMap::GEMStripMap>>();
}
