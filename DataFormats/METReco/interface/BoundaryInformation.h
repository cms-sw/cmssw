#ifndef BOUNDARYINFORMATION_H_
#define BOUNDARYINFORMATION_H_

// system include files

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

using namespace edm;
using namespace std;

class BoundaryInformation {
   public:
      BoundaryInformation() {
         recHits = vector<EcalRecHit> ();
         detIds = vector<DetId> ();
         channelStatus = vector<int> ();
         boundaryEnergy = 0.;
         boundaryET = 0.;
         subdet = EcalSubdetector();
         nextToBorder = false;
      }
      ;
      vector<EcalRecHit> recHits;
      vector<DetId> detIds;
      vector<int> channelStatus;
      double boundaryEnergy;
      double boundaryET;
      EcalSubdetector subdet;
      bool nextToBorder;
};

#endif /*BOUNDARYINFORMATION_H_*/
