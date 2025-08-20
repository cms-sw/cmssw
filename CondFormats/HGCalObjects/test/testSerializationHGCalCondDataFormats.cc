#include "CondFormats/Serialization/interface/Test.h"
#include "CondFormats/HGCalObjects/src/headers.h"

int main() {
  //dense indexers
  testSerialization<HGCalDenseIndexerBase>();
  testSerialization<HGCalMappingCellIndexer>();
  testSerialization<HGCalFEDReadoutSequence>();
  testSerialization<HGCalMappingModuleIndexer>();

  return 0;
}
