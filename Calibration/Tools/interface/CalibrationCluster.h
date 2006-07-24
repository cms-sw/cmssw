#ifndef CALIBRATION_CLUSTER
#define CALIBRATION_CLUSTER
//
// Owns map to be calibrated and calibration clusters
// 
// Author:  Lorenzo AGOSTINO

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include<vector>
#include<map>

using namespace std;

class CalibrationCluster{

public:
typedef map<EBDetId,unsigned int> CalibMap;
typedef CalibMap::value_type pippo;



CalibrationCluster();
~CalibrationCluster();

CalibMap getMap(int, int, int, int);
vector<EBDetId> get5x5Id(EBDetId const &);
vector<EBDetId> get3x3Id(EBDetId const &);
vector<float>   getEnergyVector(const EBRecHitCollection* ,CalibMap &, vector<EBDetId> &, float &, int &);

private:

vector<EBDetId> Xtals5x5;
vector<EBDetId> Xtals3x3;
vector<float> energyVector;
CalibMap calibRegion;
};

#endif
