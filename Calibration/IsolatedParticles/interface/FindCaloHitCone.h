#ifndef CalibrationIsolatedParticlesFindCaloHitCone_h
#define CalibrationIsolatedParticlesFindCaloHitCone_h

// system include files
#include <memory>
#include <cmath>
#include <string>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include <cmath>

namespace spr {

  // One Hit Collection
  template <typename T>
  std::vector<typename T::const_iterator> findHitCone(const CaloGeometry* geo, edm::Handle<T>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom);
 
  // Two Hit Collections - needed for looping over Ecal Endcap/Barrel Hits
  template <typename T>
  std::vector<typename T::const_iterator> findHitCone(const CaloGeometry* geo, edm::Handle<T>& barrelhits, edm::Handle<T>& endcaphits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom);
  
  //Ecal Endcap OR Barrel RecHits
  std::vector<EcalRecHitCollection::const_iterator> findCone(const CaloGeometry* geo, edm::Handle<EcalRecHitCollection>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR);

  // Ecal Endcap AND Barrel RecHits
  std::vector<EcalRecHitCollection::const_iterator> findCone(const CaloGeometry* geo, edm::Handle<EcalRecHitCollection>& barrelhits, edm::Handle<EcalRecHitCollection>& endcaphits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom);

  //HBHE RecHits
  std::vector<HBHERecHitCollection::const_iterator> findCone(const CaloGeometry* geo, edm::Handle<HBHERecHitCollection>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom);
  
  // PCalo SimHits
  std::vector<edm::PCaloHitContainer::const_iterator> findCone(const CaloGeometry* geo, edm::Handle<edm::PCaloHitContainer>& hits, const GlobalPoint& hpoint1, const GlobalPoint& point1, double dR, const GlobalVector& trackMom);

}

#include "FindCaloHitCone.icc"

#endif
