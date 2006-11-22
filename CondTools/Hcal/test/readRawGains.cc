#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>

#include "CondTools/Hcal/interface/HcalDbXml.h"
#include "CondFormats/HcalObjects/interface/HcalRawGains.h"

bool hasKey (const std::string& line, const std::string& key) {
  return line.find ("<"+key+">") != std::string::npos;
}

double getValue (const std::string& line) {
  unsigned pos1 = line.find (">");
  unsigned pos2 = line.find ("</");
  if (pos1 != std::string::npos && pos2 != std::string::npos) {
    std::string valueStr (line, pos1+1, pos2-pos1);
    return atof (valueStr.c_str());
  }
  std::cerr << "Can not decode line: " << line << std::endl;
  return 0;
}

int getIntValue (const std::string& line) {
  unsigned pos1 = line.find (">");
  unsigned pos2 = line.find ("</");
  if (pos1 != std::string::npos && pos2 != std::string::npos) {
    std::string valueStr (line, pos1+1, pos2-pos1);
    return atoi (valueStr.c_str());
  }
  std::cerr << "Can not decode line: " << line << std::endl;
  return 0;
}

int main (int argn, char* argv []) {
  std::ifstream inStream (argv[1]);
  std::ofstream outStream (argv[2]);
  HcalRawGains gains;
  float gain = 0;
  int eta = 0;
  int phi = 0;
  int depth = 0;
  int z = 0;
  char buffer [1024];
  while (inStream.getline (buffer, 1024)) {
    std::string line (buffer);
    if (hasKey (line, "Z")) z = getIntValue (line);
    else if (hasKey (line, "PHI")) phi = getIntValue (line);
    else if (hasKey (line, "ETA")) eta = getIntValue (line);
    else if (hasKey (line, "DEPTH")) depth = getIntValue (line);
    else if (hasKey (line, "COEFFICIENT")) gain = getValue (line);
    else if (hasKey (line, "PHI")) phi = getIntValue (line);
    else if (hasKey (line, "PHI")) phi = getIntValue (line);
    else if (hasKey (line, "/DATA")) { // flush data
      HcalDetId id (HcalEndcap, z*eta, phi, depth);
      HcalRawGain newItem (id.rawId(), gain, 0., 0., HcalRawGain::GOOD);
      gains.addValues (id, newItem);
      
    }
  }
  gains.sort ();
  HcalDbXml::dumpObject (outStream, 1, 1, 0xfffffffful, "he_signal_leveling_from_Petr", gains);
  return 0;
}

