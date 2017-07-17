#ifndef PFClusterShapeProducer_PFClusterWidthAlgo_H
#define PFClusterShapeProducer_PFClusterWidthAlgo_H
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class PFClusterWidthAlgo
{
 public:
  //constructor
  PFClusterWidthAlgo(const std::vector<const reco::PFCluster* >& pfclust);

  
  //destructor
  ~PFClusterWidthAlgo();

  inline double pflowPhiWidth() const {return phiWidth_;}
  inline double pflowEtaWidth() const {return etaWidth_;}
  inline double pflowSigmaEtaEta() const {return sigmaEtaEta_;}

 private:
  

  double phiWidth_;
  double etaWidth_;
  double sigmaEtaEta_;

};

#endif
