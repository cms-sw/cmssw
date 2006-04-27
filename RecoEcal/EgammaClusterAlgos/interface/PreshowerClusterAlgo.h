#ifndef RecoEcal_EgammaClusterAlgos_PreshowerClusterAlgo_h
#define RecoEcal_EgammaClusterAlgos_PreshowerClusterAlgo_h

#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h" // <===== Still does not exist!
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

class PreshowerClusterAlgo {

 public:

  PreshowerClusterAlgo() : 
  PreshStripEnergyCut_(0.), PreshClusterEnergyCut_(0.), PreshSeededNstr_(15)
  {}

   PreshowerClusterAlgo(double StripEnergyCut, double ClusterEnergyCut, int NStripCut) :
   PreshStripEnergyCut_(StripEnergyCut), PreshClusterEnergyCut_(ClusterEnergyCut), PreshSeededNstr_(NStripCut)
   {}

   ~PreshowerClusterAlgo();

   reco::PreshowerCluster makeOneCluster(ESDetId strip,  edm::ESHandle<CaloTopology> theCaloTopology);
   void PreshHitsInit(const EcalRecHitCollection& rechits);

 private:
  
   double PreshStripEnergyCut_;
   double PreshClusterEnergyCut_;
   int PreshSeededNstr_;

   std::map< ESDetId, std::pair<EcalRecHit, bool> >  rhits_presh;

};
#endif

