#ifndef PFClusterShapeProducer_PFClusterWidthAlgo_H
#define PFClusterShapeProducer_PFClusterWidthAlgo_H
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"


class PFClusterWidthAlgo
{
 public:
  //constructor
  PFClusterWidthAlgo(const std::vector<reco::PFCluster>& pfclust);
  
  //destructor
  ~PFClusterWidthAlgo();

  double pflowPhiWidth(){return phiWidth_;}
  double pflowEtaWidth(){return etaWidth_;}
  double pflowSigmaEtaEta(){return sigmaEtaEta_;}

 private:
  

  double phiWidth_;
  double etaWidth_;
  double sigmaEtaEta_;

};

#endif
