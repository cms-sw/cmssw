#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/BeamSpotObjects/src/headers.h"

int main() {
  testSerialization<BeamSpotObjects>();
  testSerialization<BeamSpotOnlineObjects>();
  testSerialization<SimBeamSpotObjects>();
  testSerialization<SimBeamSpotHLLHCObjects>();

  return 0;
}
