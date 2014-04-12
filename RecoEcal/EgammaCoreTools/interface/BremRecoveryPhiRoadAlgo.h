#ifndef RecoEcal_EgammaCoreTools_BremRecoveryPhiRoadAlgo_h
#define RecoEcal_EgammaCoreTools_BremRecoveryPhiRoadAlgo_h

/** \class BremRecoveryPhiRoadAlgo
 *  
 * calculates the optimal phi road length for the
 * ecal barrel or endcap. 
 *
 */

#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class BremRecoveryPhiRoadAlgo {

   public:
      BremRecoveryPhiRoadAlgo(const edm::ParameterSet& pset);
      ~BremRecoveryPhiRoadAlgo() {}

      int barrelPhiRoad(double et);
      double endcapPhiRoad(double energy);

   private:
      // parameters for EB
      // if (et < etVec[i]) use cryVec_[i]
      std::vector<double> etVec_;
      std::vector<int> cryVec_;
      int cryMin_;

      // parameters for EE
      // phi road = (a_ / (b_ + energy)) + c
      double a_;
      double b_;
      double c_;

};

#endif

