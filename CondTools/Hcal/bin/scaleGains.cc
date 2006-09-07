#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"


int main (int argn, char* argv []) {
  if (argn < 3) {
    std::cerr << "Use: " << argv[0] << " <gains to scale (.txt)> <result (.txt)> <scale factor>" << std::endl;
    return 1;
  }
  std::ifstream inStream (argv[1]);
  std::ofstream outStream (argv[2]);
  double scale = atof (argv[3]);
  HcalGains gainsIn;
  HcalDbASCIIIO::getObject (inStream, &gainsIn);
  gainsIn.sort();
  HcalGains gainsOut;
  std::vector<DetId> channels = gainsIn.getAllChannels ();
  for (unsigned i = 0; i < channels.size(); i++) {
    DetId id = channels[i];
    gainsOut.addValue (id, 
		       gainsIn.getValue (id, 0) * scale, gainsIn.getValue (id, 1) * scale, 
		       gainsIn.getValue (id, 2) * scale, gainsIn.getValue (id, 3) * scale);
  }
  gainsOut.sort();
  HcalDbASCIIIO::dumpObject (outStream, gainsOut);
  return 0;
}

