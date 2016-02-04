#ifndef RecoEcal_EgammaClusterAlgos_ExampleClusterAlgo_h
#define RecoEcal_EgammaClusterAlgos_ExampleClusterAlgo_h

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class ExampleClusterAlgo {

 public:
   ExampleClusterAlgo();
   ExampleClusterAlgo(double energyCut, int nXtalCut);
   ~ExampleClusterAlgo();

   void setEnergyCut(double value) { energyCut_ = value;}
   void setNXtalCut(int value) { nXtalCut_ = value;}

   reco::BasicCluster            makeOneCluster();
   reco::BasicClusterCollection  makeClusters(const EcalRecHitCollection& rechits);

 private:
   double energyCut_;
   int    nXtalCut_;
};
#endif
