#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/PPSObjects/src/headers.h"

int main() {
  testSerialization<PPSAssociationCuts>();
  testSerialization<PPSAssociationCuts::CutsPerArm>();
  testSerialization<std::map<unsigned int, PPSAssociationCuts::CutsPerArm>>();
  testSerialization<std::vector<std::string>>();
}