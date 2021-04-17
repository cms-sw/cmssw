#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/GBRForest/src/headers.h"

int main() {
  testSerialization<GBRForest>();
  testSerialization<GBRForest2D>();
  testSerialization<GBRForestD>();
  testSerialization<GBRTree>();
  testSerialization<GBRTree2D>();
  testSerialization<GBRTreeD>();
  testSerialization<std::vector<GBRTree2D>>();
  testSerialization<std::vector<GBRTree>>();
  testSerialization<std::vector<GBRTreeD>>();

  return 0;
}
