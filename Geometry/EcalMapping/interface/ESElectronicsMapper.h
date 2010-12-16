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

class ESElectronicsMapper {

 public:

  ESElectronicsMapper(const edm::ParameterSet& ps);
  ~ESElectronicsMapper() {};

  int getFED(const ESDetId& id);
  int getFED(int zside, int plane, int x, int y);
  std::vector<int> GetListofFEDs(const std::vector<int> eeFEDs) const ;
  void GetListofFEDs(std::vector<int> eeFEDs, std::vector<int> & esFEDs) const ;

 private:

  edm::FileInPath lookup_;

  int fed_[2][2][40][40];
  std::map < int, std::vector<int>  > ee_es_map_;

  

};

#endif
