#include "CondFormats/Serialization/interface/Test.h"
#include "CondFormats/GEMObjects/src/headers.h"

int main()
{
  testSerialization<GEMEMap>();
  testSerialization<GEMEMap::GEMEMapItem>();
  testSerialization<std::vector<GEMEMap::GEMEMapItem>>();
  testSerialization<GEMEMap::GEMVFatMaptype>();
  testSerialization<std::vector<GEMEMap::GEMVFatMaptype>>();
  testSerialization<GEMEMap::GEMVFatMapInPos>();
  testSerialization<std::vector<GEMEMap::GEMVFatMapInPos>>();

}
