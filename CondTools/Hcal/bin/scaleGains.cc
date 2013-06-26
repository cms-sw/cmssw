#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"


int main (int argn, char* argv []) {
  if (argn < 3) {
    std::cerr << "Use: " << argv[0] << " <gains to scale (.txt)> <result (.txt)> <scale factor>" << std::endl;
    return 1;
  }
  HcalTopology topo(HcalTopologyMode::LHC,2,3);

  std::ifstream inStream (argv[1]);
  std::ofstream outStream (argv[2]);
  double scale = atof (argv[3]);
  HcalGains gainsIn(&topo);;
  HcalDbASCIIIO::getObject (inStream, &gainsIn);
  HcalGains gainsOut(&topo);;
  std::vector<DetId> channels = gainsIn.getAllChannels ();
  for (unsigned i = 0; i < channels.size(); i++) {
    DetId id = channels[i];
    HcalGain item(id, gainsIn.getValues(id)->getValue(0) * scale, gainsIn.getValues(id)->getValue(1) * scale, 
		  gainsIn.getValues(id)->getValue(2) * scale, gainsIn.getValues(id)->getValue(3) * scale);
    gainsOut.addValues(item);
  }
  HcalDbASCIIIO::dumpObject (outStream, gainsOut);
  return 0;
}

