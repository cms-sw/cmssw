#ifndef RecoEcal_EgammaCoreTools_BremRecoveryPhiRoadAlgo_h
#define RecoEcal_EgammaCoreTools_BremRecoveryPhiRoadAlgo_h

/** \class BremRecoveryPhiRoadAlgo
 *  
 * calculates the optimal phi road length for the
 * ecal barrel or endcap. 
 *
 */

class BremRecoveryPhiRoadAlgo {

   public:
      BremRecoveryPhiRoadAlgo() {};
      ~BremRecoveryPhiRoadAlgo() {};
      int barrelPhiRoad(double et);
      double endcapPhiRoad(double energy);
};

#endif

