#include <iostream>

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "RecoEcal/EgammaCoreTools/interface/SuperClusterShapeAlgo.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"

SuperClusterShapeAlgo::SuperClusterShapeAlgo(const EcalRecHitCollection* hits,
					     const CaloSubdetectorGeometry* geo) : 
  recHits_(hits), geometry_(geo) { }

void SuperClusterShapeAlgo::Calculate_Covariances(const reco::SuperCluster &passedCluster)
{
  double numeratorEtaWidth = 0;
  double numeratorPhiWidth = 0;

  double scEnergy = passedCluster.energy();
  double denominator = scEnergy;

  double scEta    = passedCluster.position().eta();
  double scPhi    = passedCluster.position().phi();

  const std::vector<std::pair<DetId,float> > &detId = passedCluster.hitsAndFractions();
  // Loop over recHits associated with the given SuperCluster
  for(std::vector<std::pair<DetId,float> >::const_iterator hit = detId.begin();
      hit != detId.end(); ++hit) {
    EcalRecHitCollection::const_iterator rHit = recHits_->find((*hit).first);
 //FIXME: THIS IS JUST A WORKAROUND A FIX SHOULD BE APPLIED  
 if(rHit == recHits_->end()) {
    continue; 
   }
    const CaloCellGeometry *this_cell = geometry_->getGeometry(rHit->id());
    if ( this_cell == 0 ) {
      //edm::LogInfo("SuperClusterShapeAlgo") << "pointer to the cell in Calculate_Covariances is NULL!";
      continue;
    }
    GlobalPoint position = this_cell->getPosition();
    //take into account energy fractions
    double energyHit = rHit->energy()*hit->second;
    
    //form differences
    double dPhi = position.phi() - scPhi;
    if (dPhi > + Geom::pi()) { dPhi = Geom::twoPi() - dPhi; }
    if (dPhi < - Geom::pi()) { dPhi = Geom::twoPi() + dPhi; }

    double dEta = position.eta() - scEta;

    if ( energyHit > 0 ) {
      numeratorEtaWidth += energyHit * dEta * dEta;
      numeratorPhiWidth += energyHit * dPhi * dPhi;
    }
      
    etaWidth_ = sqrt(numeratorEtaWidth / denominator);
    phiWidth_ = sqrt(numeratorPhiWidth / denominator);
  }
}
