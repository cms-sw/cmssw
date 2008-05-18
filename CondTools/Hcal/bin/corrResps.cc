#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"


int main (int argn, char* argv []) {
  if (argn < 3) {
    std::cerr << "Use: " << argv[0] << " <RespCorrs to scale (.txt)> <resultRespCorrs (.txt)> <correction-respcorr (.txt)>" << std::endl;
    return 1;
  }
  std::ifstream inStream (argv[1]);
  std::ofstream outStream (argv[2]);
  std::ifstream inCorr (argv[3]);
  HcalRespCorrs respIn;
  HcalDbASCIIIO::getObject (inStream, &respIn);
  HcalRespCorrs corrsIn;
  HcalDbASCIIIO::getObject (inCorr, &corrsIn);

  HcalRespCorrs respOut;
  std::vector<DetId> channels = respIn.getAllChannels ();
  for (unsigned i = 0; i < channels.size(); i++) {
    DetId id = channels[i];
    float scale = corrsIn.getValues(id)->getValue();
    HcalRespCorr item (id, respIn.getValues(id)->getValue() * scale);
    respOut.addValues(item);
  }
  HcalDbASCIIIO::dumpObject (outStream, respOut);
  return 0;
}

