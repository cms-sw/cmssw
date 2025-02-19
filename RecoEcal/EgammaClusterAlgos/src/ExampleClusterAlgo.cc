#include "RecoEcal/EgammaClusterAlgos/interface/ExampleClusterAlgo.h"

ExampleClusterAlgo::ExampleClusterAlgo() :
  energyCut_(0.), nXtalCut_(-1) {
}


ExampleClusterAlgo::ExampleClusterAlgo(double energyCut, int nXtalCut) {
  energyCut_ = energyCut;
  nXtalCut_ = nXtalCut;
}

ExampleClusterAlgo::~ExampleClusterAlgo() {

}


reco::BasicCluster
ExampleClusterAlgo::makeOneCluster() {

  return reco::BasicCluster();

}

reco::BasicClusterCollection
ExampleClusterAlgo::makeClusters(const EcalRecHitCollection& rechits) {

  return reco::BasicClusterCollection();

}

