#ifndef RecoECAL_ECALClusters_EgammaSCEnergyCorrectionAlgo_h_
#define RecoECAL_ECALClusters_EgammaSCEnergyCorrectionAlgo_h_

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <map>
#include <string>

class EgammaSCEnergyCorrectionAlgo
{
  public:
    EgammaSCEnergyCorrectionAlgo(double noise);
    ~EgammaSCEnergyCorrectionAlgo();
    reco::SuperCluster applyCorrection(const reco::SuperCluster &cl, const EcalRecHitCollection &rhc, reco::AlgoId theAlgo);
  
  private:    
    float fNCrystals(int nCry, reco::AlgoId theAlgo, EcalSubdetector theBase);
    int nCrystalsGT2Sigma(const reco::BasicCluster &seed);
	double sigmaElectronicNoise_;
    std::map<DetId, EcalRecHit> *recHits_m;
};


#endif /*RecoECAL_ECALClusters_EgammaSCEnergyCorrectionAlgo_h_*/
