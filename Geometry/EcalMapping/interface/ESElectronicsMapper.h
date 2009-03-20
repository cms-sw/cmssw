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
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"

class ESElectronicsMapper {

 public:

  ESElectronicsMapper(const edm::ParameterSet& ps);
  ~ESElectronicsMapper() {};

  int getFED(const ESDetId& id);
  int getFED(int zside, int plane, int x, int y);
  std::vector<int> GetListofFEDs(const EcalEtaPhiRegion region) const ;
  void GetListofFEDs(const EcalEtaPhiRegion region, std::vector<int> & FEDs) const ;
  void findXY(const int plane, const double eta, const double phi, int &row, int &col) const;

 private:

  edm::FileInPath lookup_;

  int fed_[2][2][40][40];

};

#endif
