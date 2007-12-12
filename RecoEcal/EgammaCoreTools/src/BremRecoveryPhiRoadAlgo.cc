
#include "RecoEcal/EgammaCoreTools/interface/BremRecoveryPhiRoadAlgo.h"

int BremRecoveryPhiRoadAlgo::barrelPhiRoad(double et)
{

   // 
   // Take as input the ET in 5x5 crystals
   // and compute the optimal phi road 
   // as a number of crystals

   if (et < 5) return 16;
   else if (et < 10) return 13;
   else if (et < 15) return 11;
   else if (et < 20) return 10;
   else if (et < 30) return 9;
   else if (et < 40) return 8;
   else if (et < 45) return 7;
   else if (et < 55) return 6;
   else if (et < 135) return 5;
   else if (et < 195) return 4;
   else if (et < 225) return 3;
   else return 2;

}

double BremRecoveryPhiRoadAlgo::endcapPhiRoad(double energy)
{

   //
   // Take as input the energy in the seed BasicCluster
   // and return the optimal phi road
   // length in radians

   double A = 47.85;
   double B = 108.8;
   double C = 0.1201;
   return ((A / (energy + B)) + C);

}

