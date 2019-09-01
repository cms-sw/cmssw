#ifndef BOUNDARYINFORMATION_H_
#define BOUNDARYINFORMATION_H_

// system include files
#include <vector>

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

//using namespace edm;
//using namespace std;

class BoundaryInformation {
public:
  BoundaryInformation() {
    recHits = std::vector<EcalRecHit>();
    detIds = std::vector<DetId>();
    channelStatus = std::vector<int>();
    boundaryEnergy = 0.;
    boundaryET = 0.;
    subdet = EcalSubdetector();
    nextToBorder = false;
  };
  std::vector<EcalRecHit> recHits;
  std::vector<DetId> detIds;
  std::vector<int> channelStatus;
  double boundaryEnergy;
  double boundaryET;
  EcalSubdetector subdet;
  bool nextToBorder;
};

#endif /*BOUNDARYINFORMATION_H_*/
