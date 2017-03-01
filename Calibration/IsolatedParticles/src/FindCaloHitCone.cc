#include "Calibration/IsolatedParticles/interface/CaloConstants.h"
#include "Calibration/IsolatedParticles/interface/FindCaloHitCone.h"
#include "Calibration/IsolatedParticles/interface/FindDistCone.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include <iostream>

namespace spr {


  //Ecal Endcap OR Barrel RecHits
  std::vector<EcalRecHitCollection::const_iterator> findCone(const CaloGeometry* geo, edm::Handle<EcalRecHitCollection>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, bool debug) {
  
    std::vector<EcalRecHitCollection::const_iterator> hit;
  
    for (EcalRecHitCollection::const_iterator j=hits->begin(); 
	 j!=hits->end(); j++) {

      bool keepHit = false;
    
      if (j->id().subdetId() == EcalEndcap) {
	EEDetId EEid = EEDetId(j->id());
	const GlobalPoint rechitPoint = geo->getPosition(EEid);
	if (spr::getDistInPlaneTrackDir(point1, trackMom, rechitPoint, debug) < dR) keepHit = true;
      } else if (j->id().subdetId() == EcalBarrel) {
	EBDetId EBid = EBDetId(j->id());
	const GlobalPoint rechitPoint = geo->getPosition(EBid);
	if (spr::getDistInPlaneTrackDir(point1, trackMom, rechitPoint, debug) < dR) keepHit = true;
      }

      if (keepHit) hit.push_back(j);
    }
    return hit;
  }

  // Ecal Endcap AND Barrel RecHits
  std::vector<EcalRecHitCollection::const_iterator> findCone(const CaloGeometry* geo, edm::Handle<EcalRecHitCollection>& barrelhits, edm::Handle<EcalRecHitCollection>& endcaphits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, bool debug) {
  
    std::vector<EcalRecHitCollection::const_iterator> hit;
  
    // Only check both barrel and endcap when track is near transition
    // region: 1.479-2*0.087 < trkEta < 1.479+2*0.087
  
    bool doBarrel=false, doEndcap=false;
    if ( std::abs(point1.eta()) < (spr::etaBEEcal+2*spr::deltaEta)) doBarrel=true; // 1.479+2*0.087
    if ( std::abs(point1.eta()) > (spr::etaBEEcal-2*spr::deltaEta)) doEndcap=true; // 1.479-2*0.087    
  
    if (doBarrel) {  
      for (EcalRecHitCollection::const_iterator j=barrelhits->begin(); 
	   j!=barrelhits->end(); j++) {

	bool keepHit = false;
	if (j->id().subdetId() == EcalBarrel) {
	  EBDetId EBid = EBDetId(j->id());
	  const GlobalPoint rechitPoint = geo->getPosition(EBid);
	  if (spr::getDistInPlaneTrackDir(point1, trackMom, rechitPoint, debug) < dR) keepHit = true;
	} else {
	  std::cout << "PROBLEM : Endcap RecHits in Barrel Collection!?" 
		    << std::endl;
	}
	if (keepHit) hit.push_back(j);
      }
    } // doBarrel
  
    if (doEndcap) {  
    
      for (EcalRecHitCollection::const_iterator j=endcaphits->begin(); 
	   j!=endcaphits->end(); j++) {
      
	bool keepHit = false;
      
	if (j->id().subdetId() == EcalEndcap) {
	  EEDetId EEid = EEDetId(j->id());
	  const GlobalPoint rechitPoint = geo->getPosition(EEid);
	  if (spr::getDistInPlaneTrackDir(point1, trackMom, rechitPoint, debug) < dR) keepHit = true;
	} else {
	  std::cout << "PROBLEM : Barrel RecHits in Endcap Collection!?" 
		    << std::endl;
	}
	if (keepHit) hit.push_back(j);
      }
    } // doEndcap
  
    return hit;
  }


  //HBHE RecHits
  std::vector<HBHERecHitCollection::const_iterator> findCone(const CaloGeometry* geo, edm::Handle<HBHERecHitCollection>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, bool debug) {

    std::vector<HBHERecHitCollection::const_iterator> hit;
    // Loop over Hcal RecHits
    for (HBHERecHitCollection::const_iterator j=hits->begin(); 
	 j!=hits->end(); j++) {   
      DetId detId(j->id());
      const GlobalPoint rechitPoint = geo->getPosition(detId);
      if (spr::getDistInPlaneTrackDir(hpoint1, trackMom, rechitPoint, debug) < dR) hit.push_back(j);
    }  
    return hit;
  }

  // PCalo SimHits
  std::vector<edm::PCaloHitContainer::const_iterator> findCone(const CaloGeometry* geo, edm::Handle<edm::PCaloHitContainer>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom, bool debug) {

    std::vector<edm::PCaloHitContainer::const_iterator> hit;  
    edm::PCaloHitContainer::const_iterator ihit;
    for (ihit=hits->begin(); ihit!=hits->end(); ihit++) {
      DetId detId(ihit->id());
      const GlobalPoint rechitPoint = geo->getPosition(detId);
      if (spr::getDistInPlaneTrackDir(hpoint1, trackMom, rechitPoint, debug) < dR) {
	hit.push_back(ihit);
      }
    }
    return hit;
  }

}
