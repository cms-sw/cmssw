#ifndef CALIBRATION_CLUSTER
#define CALIBRATION_CLUSTER
//
// Owns map to be calibrated and calibration clusters
//
// Author:  Lorenzo AGOSTINO

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include <vector>
#include <map>

class CalibrationCluster {
public:
  typedef std::map<EBDetId, unsigned int> CalibMap;
  typedef CalibMap::value_type pippo;

  CalibrationCluster();
  ~CalibrationCluster();

  CalibMap getMap(int, int, int, int);
  std::vector<EBDetId> get5x5Id(EBDetId const &);
  std::vector<EBDetId> get3x3Id(EBDetId const &);
  std::vector<float> getEnergyVector(const EBRecHitCollection *, CalibMap &, std::vector<EBDetId> &, float &, int &);

private:
  std::vector<EBDetId> Xtals5x5;
  std::vector<EBDetId> Xtals3x3;
  std::vector<float> energyVector;
  CalibMap calibRegion;
};

#endif
