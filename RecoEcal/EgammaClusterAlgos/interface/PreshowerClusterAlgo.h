#ifndef RecoEcal_EgammaClusterAlgos_PreshowerClusterAlgo_h
#define RecoEcal_EgammaClusterAlgos_PreshowerClusterAlgo_h
//
// $Id: $
//

#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
//#include "RecoEcal/EgammaClusterAlgos/interface/LogPositionCalc.h"



class PreshowerClusterAlgo {

 public:

  typedef math::XYZPoint Point;

  PreshowerClusterAlgo() : 
  PreshStripEnergyCut_(0.), PreshClusterEnergyCut_(0.), PreshSeededNstr_(15)
  {}

   PreshowerClusterAlgo(double StripEnergyCut, double ClusterEnergyCut, int NStripCut) :
   PreshStripEnergyCut_(StripEnergyCut), PreshClusterEnergyCut_(ClusterEnergyCut), PreshSeededNstr_(NStripCut)
   {}

   ~PreshowerClusterAlgo()
    {
    };

   reco::PreshowerCluster makeOneCluster(ESDetId strip,  edm::ESHandle<CaloTopology> theCaloTopology, edm::ESHandle<CaloGeometry> geometry_h);
   void PreshHitsInit(const EcalRecHitCollection& rechits);

 private:
  
   double PreshStripEnergyCut_;
   double PreshClusterEnergyCut_;
   int PreshSeededNstr_;

   std::map< ESDetId, std::pair<EcalRecHit, bool> >  rhits_presh;

   friend Point getECALposition(std::vector<reco::EcalRecHitData> recHits,const CaloSubdetectorGeometry );
   //Position determination

};
#endif

