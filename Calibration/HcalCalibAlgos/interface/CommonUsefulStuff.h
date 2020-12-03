#ifndef Calibration_HcalCalibALgos_CommonUsefulStuff_h
#define Calibration_HcalCalibALgos_CommonUsefulStuff_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/DeltaR.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TString.h"

/* getDist functions by Jim:
/uscms/home/jhirsch/IsoTracks_314/CMSSW_3_1_4/src/JetMETCorrections/IsolatedParticles/interface/FindCaloHit.icc
*/

struct MaxHit_struct {
  int iphihitm;
  int ietahitm;
  int depthhit;
  float hitenergy;
  float dr;
  GlobalPoint posMax;
  MaxHit_struct() : iphihitm(0), ietahitm(0), depthhit(0), hitenergy(-100), dr(0) {}
};

inline double getDistInPlaneSimple(const GlobalPoint caloPoint, const GlobalPoint rechitPoint) {
  // Simplified version of getDistInPlane
  // Assume track direction is origin -> point of hcal intersection

  const GlobalVector caloIntersectVector(caloPoint.x(), caloPoint.y(), caloPoint.z());

  const GlobalVector caloIntersectUnitVector = caloIntersectVector.unit();

  const GlobalVector rechitVector(rechitPoint.x(), rechitPoint.y(), rechitPoint.z());

  const GlobalVector rechitUnitVector = rechitVector.unit();
  double dotprod = caloIntersectUnitVector.dot(rechitUnitVector);
  double rechitdist = caloIntersectVector.mag() / dotprod;

  const GlobalVector effectiveRechitVector = rechitdist * rechitUnitVector;
  const GlobalPoint effectiveRechitPoint(
      effectiveRechitVector.x(), effectiveRechitVector.y(), effectiveRechitVector.z());

  GlobalVector distance_vector = effectiveRechitPoint - caloPoint;

  if (dotprod > 0.) {
    return distance_vector.mag();
  } else {
    return 999999.;
  }
}

inline double getDistInPlaneTrackDir(const GlobalPoint caloPoint,
                                     const GlobalVector caloVector,
                                     const GlobalPoint rechitPoint) {
  // Simplified version of getDistInPlane : no cone "within" Hcal, but
  // don't assume track direction is origin -> point of hcal
  // intersection.
  const GlobalVector caloIntersectVector(caloPoint.x(), caloPoint.y(),
                                         caloPoint.z());  //p

  const GlobalVector caloUnitVector = caloVector.unit();
  const GlobalVector rechitVector(rechitPoint.x(), rechitPoint.y(), rechitPoint.z());
  const GlobalVector rechitUnitVector = rechitVector.unit();
  double dotprod_denominator = caloUnitVector.dot(rechitUnitVector);
  double dotprod_numerator = caloUnitVector.dot(caloIntersectVector);
  double rechitdist = dotprod_numerator / dotprod_denominator;
  const GlobalVector effectiveRechitVector = rechitdist * rechitUnitVector;
  const GlobalPoint effectiveRechitPoint(
      effectiveRechitVector.x(), effectiveRechitVector.y(), effectiveRechitVector.z());
  GlobalVector distance_vector = effectiveRechitPoint - caloPoint;
  if (dotprod_denominator > 0. && dotprod_numerator > 0.) {
    return distance_vector.mag();
  } else {
    return 999999.;
  }
}

inline double getDistInPlane(const GlobalVector trackDirection,
                             const GlobalPoint caloPoint,
                             const GlobalPoint rechitPoint,
                             double coneHeight) {
  // The iso track candidate hits the Calo (Ecal or Hcal) at "caloPoint"
  // with direction "trackDirection".

  // "rechitPoint" is the position of the rechit.  We only care about
  // the direction of the rechit.

  // Consider the rechitPoint as characterized by angles theta and phi
  // wrt the origin which points at the calo cell of the rechit.  In
  // some sense the distance along the line theta/phi is arbitrary. A
  // simplified choice might be to put the rechit at the surface of
  // the Hcal.  Our approach is to see whether/where this line
  // intersects a cone of height "coneHeight" with vertex at caloPoint
  // aligned with trackDirection.
  // To that end, this function returns the distance between the
  // center of the base of the cone and the intersection of the rechit
  // line and the plane that contains the base of the cone.  This
  // distance is compared with the radius of the cone outside this
  // function.

  // Make vector of length cone height along track direction
  const GlobalVector heightVector = trackDirection * coneHeight;

  // Make vector from origin to point where iso track intersects the
  // calorimeter.
  const GlobalVector caloIntersectVector(caloPoint.x(), caloPoint.y(), caloPoint.z());

  // Make vector from origin to point in center of base of cone
  const GlobalVector coneBaseVector = heightVector + caloIntersectVector;

  // Make point in center of base of cone
  const GlobalPoint coneBasePoint(coneBaseVector.x(), coneBaseVector.y(), coneBaseVector.z());

  // Make unit vector pointing at rechit.
  const GlobalVector rechitVector(rechitPoint.x(), rechitPoint.y(), rechitPoint.z());
  const GlobalVector rechitDirection = rechitVector.unit();

  // Find distance "r" along "rechit line" (with angles theta2 and
  // phi2) where line intersects plane defined by base of cone.

  // Definition plane of that contains base of cone:
  // trackDirection.x() (x - coneBaseVector.x()) +
  // trackDirection.y() (y - coneBaseVector.y()) +
  // trackDirection.z() (z - coneBaseVector.z()) = 0

  // Point P_{rh} where rechit line intersects plane:
  // (rechitdist sin(theta2) cos(phi2),
  //  rechitdist sin(theta2) cos(phi2),
  //  rechitdist cos(theta2))

  // Substitute P_{rh} into equation for plane and solve for rechitdist.
  // rechitDist turns out to be the ratio of dot products:

  double rechitdist = trackDirection.dot(coneBaseVector) / trackDirection.dot(rechitDirection);

  // Now find distance between point at center of cone base and point
  // where the "rechit line" intersects the plane defined by the base
  // of the cone; i.e. the effectiveRecHitPoint.
  const GlobalVector effectiveRechitVector = rechitdist * rechitDirection;
  const GlobalPoint effectiveRechitPoint(
      effectiveRechitVector.x(), effectiveRechitVector.y(), effectiveRechitVector.z());

  GlobalVector distance_vector = effectiveRechitPoint - coneBasePoint;
  return distance_vector.mag();
}

/*  Function to calculate Ecal Energy in Cone (given in cm) */
inline double ecalEnergyInCone(const GlobalPoint center,
                               double radius,
                               const EcalRecHitCollection ecalCol,
                               const CaloGeometry* geo) {
  double eECALcone = 0;
  std::vector<int> usedHitsEcal;
  usedHitsEcal.clear();

  for (std::vector<EcalRecHit>::const_iterator ehit = ecalCol.begin(); ehit != ecalCol.end(); ehit++) {
    //This is a precaution for the case when hitCollection contains duplicats.
    bool hitIsUsed = false;
    int hitHashedIndex = -10000;
    if (ehit->id().subdetId() == EcalBarrel) {
      EBDetId did(ehit->id());
      hitHashedIndex = did.hashedIndex();
    }
    if (ehit->id().subdetId() == EcalEndcap) {
      EEDetId did(ehit->id());
      hitHashedIndex = did.hashedIndex();
    }
    for (uint32_t i = 0; i < usedHitsEcal.size(); i++) {
      if (usedHitsEcal[i] == hitHashedIndex)
        hitIsUsed = true;
    }
    if (hitIsUsed)
      continue;
    usedHitsEcal.push_back(hitHashedIndex);
    // -----------------------------------------------

    const GlobalPoint& pos = geo->getPosition((*ehit).detid());

    if (getDistInPlaneSimple(center, pos) < radius) {
      eECALcone += ehit->energy();
    }
  }
  return eECALcone;
}

/*  This is another version of ecalEnergy calculation using the getDistInPlaneTrackDir()  */
inline double ecalEnergyInCone(const GlobalVector trackMom,
                               const GlobalPoint center,
                               double radius,
                               const EcalRecHitCollection ecalCol,
                               const CaloGeometry* geo) {
  double eECALcone = 0;
  std::vector<int> usedHitsEcal;
  usedHitsEcal.clear();
  for (std::vector<EcalRecHit>::const_iterator ehit = ecalCol.begin(); ehit != ecalCol.end(); ehit++) {
    //This is a precaution for the case when hitCollection contains duplicats.
    bool hitIsUsed = false;
    int hitHashedIndex = -10000;
    if (ehit->id().subdetId() == EcalBarrel) {
      EBDetId did(ehit->id());
      hitHashedIndex = did.hashedIndex();
    }
    if (ehit->id().subdetId() == EcalEndcap) {
      EEDetId did(ehit->id());
      hitHashedIndex = did.hashedIndex();
    }
    for (uint32_t i = 0; i < usedHitsEcal.size(); i++) {
      if (usedHitsEcal[i] == hitHashedIndex)
        hitIsUsed = true;
    }
    if (hitIsUsed)
      continue;
    usedHitsEcal.push_back(hitHashedIndex);
    // -----------------------------------------------

    const GlobalPoint& pos = geo->getPosition((*ehit).detid());

    if (getDistInPlaneTrackDir(center, trackMom, pos) < radius) {
      eECALcone += ehit->energy();
    }
  }
  return eECALcone;
}

#endif
