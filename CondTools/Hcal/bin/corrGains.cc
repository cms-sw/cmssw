#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"


int main (int argn, char* argv []) {
  if (argn < 3) {
    std::cerr << "Use: " << argv[0] << " <gains to scale (.txt)> <result (.txt)> <respcorr (.txt)>" << std::endl;
    return 1;
  }
  std::ifstream inStream (argv[1]);
  std::ofstream outStream (argv[2]);
  std::ifstream inCorr (argv[3]);
  HcalTopology topo(HcalTopologyMode::LHC,2,3);
  HcalGains gainsIn(&topo);;
  HcalDbASCIIIO::getObject (inStream, &gainsIn);
  HcalRespCorrs corrsIn(&topo);;
  HcalDbASCIIIO::getObject (inCorr, &corrsIn);

  HcalGains gainsOut(&topo);;
  std::vector<DetId> channels = gainsIn.getAllChannels ();
  for (unsigned i = 0; i < channels.size(); i++) {
    DetId id = channels[i];
    float scale = 1.;
    if (corrsIn.exists(id)) scale = corrsIn.getValues(id)->getValue();
    HcalGain item(id, gainsIn.getValues(id)->getValue(0) * scale, gainsIn.getValues(id)->getValue(1) * scale, 
		  gainsIn.getValues(id)->getValue(2) * scale, gainsIn.getValues(id)->getValue(3) * scale);
    gainsOut.addValues(item);
  }
  HcalDbASCIIIO::dumpObject (outStream, gainsOut);
  return 0;
}

