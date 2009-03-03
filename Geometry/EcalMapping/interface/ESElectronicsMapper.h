#ifndef ESElectronicsMapper_H
#define ESElectronicsMapper_H

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

using namespace std;
using namespace edm;

class ESElectronicsMapper {

 public:

  ESElectronicsMapper(const ParameterSet& ps);
  ~ESElectronicsMapper() {};

  int getFED(const ESDetId& id);
  int getFED(int zside, int plane, int x, int y);

 private:

  FileInPath lookup_;

  int fed_[2][2][40][40];

};

#endif
