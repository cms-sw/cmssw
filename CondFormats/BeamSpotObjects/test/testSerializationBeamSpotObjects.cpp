#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/BeamSpotObjects/src/headers.h"

int main() {
  testSerialization<BeamSpotObjects>();
  testSerialization<SimBeamSpotObjects>();
  testSerialization<BeamSpotOnlineObjects>();

  return 0;
}
