#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

int main (int argn, char* argv []) {
  if (argn < 3) {
    std::cerr << "Use: " << argv[0] << " <RespCorrs to scale (.txt)> <resultRespCorrs (.txt)> <correction-respcorr (.txt)>" << std::endl;
    return 1;
  }
  HcalTopology topo(HcalTopologyMode::LHC,2,3);
  std::ifstream inStream (argv[1]);
  std::ofstream outStream (argv[2]);
  std::ifstream inCorr (argv[3]);
  HcalRespCorrs respIn(&topo);
  HcalDbASCIIIO::getObject (inStream, &respIn);
  HcalRespCorrs corrsIn(&topo);
  HcalDbASCIIIO::getObject (inCorr, &corrsIn);

  HcalRespCorrs respOut(&topo);
  std::vector<DetId> channels = respIn.getAllChannels ();
  for (unsigned i = 0; i < channels.size(); i++) {
    DetId id = channels[i];
    float scale = 1.0;
    if (corrsIn.exists(id)) scale = corrsIn.getValues(id)->getValue();
    HcalRespCorr item (id, respIn.getValues(id)->getValue() * scale);
    respOut.addValues(item);
  }
  HcalDbASCIIIO::dumpObject (outStream, respOut);
  return 0;
}

