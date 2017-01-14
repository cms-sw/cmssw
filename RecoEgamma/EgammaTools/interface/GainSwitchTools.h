#ifndef RecoEgamma_EGammaTools_GainSwitchTools_h
#define RecoEgamma_EGammaTools_GainSwitchTools_h

 
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include <vector>

class DetId;
namespace reco{
  class SuperCluster;
}
class CaloTopology;

class GainSwitchTools {

public:

  //this should really live in EcalClusterTools
  static int nrCrysWithFlagsIn5x5(const DetId& id,const std::vector<int>& flags,const EcalRecHitCollection* recHits,const CaloTopology *topology);
  
  //note, right now the weights are showing the GS flags so the collections here have to be pure multifit
  static bool hasEBGainSwitch(const reco::SuperCluster& superClus,const EcalRecHitCollection* recHits);
  static bool hasEBGainSwitchIn5x5(const reco::SuperCluster& superClus,const EcalRecHitCollection* recHits,const CaloTopology *topology);
  static bool hasEBGainSwitch(const EcalRecHitCollection* recHits);
  
  static const std::vector<int> gainSwitchFlags(){return gainSwitchFlags_;}

private:
  static const std::vector<int> gainSwitchFlags_;
  
};


#endif
