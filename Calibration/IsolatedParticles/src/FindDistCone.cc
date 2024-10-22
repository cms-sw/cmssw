#include "Calibration/IsolatedParticles/interface/CaloConstants.h"
#include "Calibration/IsolatedParticles/interface/FindDistCone.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace spr {

  // Cone clustering core
  double getDistInPlaneTrackDir(const GlobalPoint& caloPoint,
                                const GlobalVector& caloVector,
                                const GlobalPoint& rechitPoint,
                                bool debug) {
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
    if (debug) {
      edm::LogVerbatim("IsoTrack") << "getDistInPlaneTrackDir: point " << caloPoint << " dirn " << caloVector
                                   << " numerator " << dotprod_numerator << " denominator " << dotprod_denominator
                                   << " distance " << distance_vector.mag();
    }
    if (dotprod_denominator > 0. && dotprod_numerator > 0.) {
      return distance_vector.mag();
    } else {
      return 999999.;
    }
  }

  // Not used, but here for reference
  double getDistInCMatEcal(double eta1, double phi1, double eta2, double phi2, bool debug) {
    double dR, Rec;
    if (fabs(eta1) < spr::etaBEEcal)
      Rec = spr::rFrontEB;
    else
      Rec = spr::zFrontEE;
    double ce1 = cosh(eta1);
    double ce2 = cosh(eta2);
    double te1 = tanh(eta1);
    double te2 = tanh(eta2);

    double z = cos(phi1 - phi2) / ce1 / ce2 + te1 * te2;
    if (z != 0)
      dR = fabs(Rec * ce1 * sqrt(1. / z / z - 1.));
    else
      dR = 999999.;
    if (debug)
      edm::LogVerbatim("IsoTrack") << "getDistInCMatEcal: between (" << eta1 << ", " << phi1 << ") and (" << eta2
                                   << ", " << phi2 << " is " << dR;
    return dR;
  }

  // Not used, but here for reference
  double getDistInCMatHcal(double eta1, double phi1, double eta2, double phi2, bool debug) {
    // Radii and eta from Geometry/HcalCommonData/data/hcalendcapalgo.xml
    // and Geometry/HcalCommonData/data/hcalbarrelalgo.xml

    double dR, Rec;
    if (fabs(eta1) < spr::etaBEHcal)
      Rec = spr::rFrontHB;
    else
      Rec = spr::zFrontHE;
    double ce1 = cosh(eta1);
    double ce2 = cosh(eta2);
    double te1 = tanh(eta1);
    double te2 = tanh(eta2);

    double z = cos(phi1 - phi2) / ce1 / ce2 + te1 * te2;
    if (z != 0)
      dR = fabs(Rec * ce1 * sqrt(1. / z / z - 1.));
    else
      dR = 999999.;
    if (debug)
      edm::LogVerbatim("IsoTrack") << "getDistInCMatHcal: between (" << eta1 << ", " << phi1 << ") and (" << eta2
                                   << ", " << phi2 << " is " << dR;
    return dR;
  }

  void getEtaPhi(HBHERecHitCollection::const_iterator hit,
                 std::vector<int>& RH_ieta,
                 std::vector<int>& RH_iphi,
                 std::vector<double>& RH_ene,
                 bool) {
    RH_ieta.push_back(hit->id().ieta());
    RH_iphi.push_back(hit->id().iphi());
    RH_ene.push_back(hit->energy());
  }

  void getEtaPhi(edm::PCaloHitContainer::const_iterator hit,
                 std::vector<int>& RH_ieta,
                 std::vector<int>& RH_iphi,
                 std::vector<double>& RH_ene,
                 bool) {
    // SimHit function not yet implemented.
    RH_ieta.push_back(-9);
    RH_iphi.push_back(-9);
    RH_ene.push_back(-9.);
  }

  void getEtaPhi(EcalRecHitCollection::const_iterator hit,
                 std::vector<int>& RH_ieta,
                 std::vector<int>& RH_iphi,
                 std::vector<double>& RH_ene,
                 bool) {
    // Ecal function not yet implemented.
    RH_ieta.push_back(-9);
    RH_iphi.push_back(-9);
    RH_ene.push_back(-9.);
  }

  void getEtaPhi(HBHERecHitCollection::const_iterator hit, int& ieta, int& iphi, bool) {
    ieta = hit->id().ieta();
    iphi = hit->id().iphi();
  }

  void getEtaPhi(edm::PCaloHitContainer::const_iterator hit, int& ieta, int& iphi, bool) {
    DetId id = DetId(hit->id());
    if (id.det() == DetId::Hcal) {
      ieta = ((HcalDetId)(hit->id())).ieta();
      iphi = ((HcalDetId)(hit->id())).iphi();
    } else if (id.det() == DetId::Ecal && id.subdetId() == EcalBarrel) {
      ieta = ((EBDetId)(id)).ieta();
      iphi = ((EBDetId)(id)).iphi();
    } else {
      ieta = 999;
      iphi = 999;
    }
  }

  void getEtaPhi(EcalRecHitCollection::const_iterator hit, int& ieta, int& iphi, bool) {
    DetId id = hit->id();
    if (id.subdetId() == EcalBarrel) {
      ieta = ((EBDetId)(id)).ieta();
      iphi = ((EBDetId)(id)).iphi();
    } else {
      ieta = 999;
      iphi = 999;
    }
  }

  double getEnergy(HBHERecHitCollection::const_iterator hit, int useRaw, bool) {
    double energy = ((useRaw == 1) ? hit->eraw() : ((useRaw == 2) ? hit->eaux() : hit->energy()));
    return energy;
  }

  double getEnergy(EcalRecHitCollection::const_iterator hit, int, bool) { return hit->energy(); }

  double getEnergy(edm::PCaloHitContainer::const_iterator hit, int, bool) {
    // This will not yet handle Ecal CaloHits!!
    double samplingWeight = 1.;
    // Hard coded sampling weights from JFH analysis of iso tracks
    // Sept 2009.
    DetId id = hit->id();
    if (id.det() == DetId::Hcal) {
      HcalDetId detId(id);
      if (detId.subdet() == HcalBarrel)
        samplingWeight = 114.1;
      else if (detId.subdet() == HcalEndcap)
        samplingWeight = 167.3;
      else {
        // ONLY protection against summing HO, HF simhits
        return 0.;
      }
    }

    return samplingWeight * hit->energy();
  }

  GlobalPoint getGpos(const CaloGeometry* geo, HBHERecHitCollection::const_iterator hit, bool) {
    DetId detId(hit->id());
    return (static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(detId)))->getPosition(detId);
  }

  GlobalPoint getGpos(const CaloGeometry* geo, edm::PCaloHitContainer::const_iterator hit, bool) {
    DetId detId(hit->id());
    GlobalPoint point = (detId.det() == DetId::Hcal)
                            ? (static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(detId)))->getPosition(detId)
                            : geo->getPosition(detId);
    return point;
  }

  GlobalPoint getGpos(const CaloGeometry* geo, EcalRecHitCollection::const_iterator hit, bool) {
    // Not tested for EcalRecHits!!
    if (hit->id().subdetId() == EcalEndcap) {
      EEDetId EEid = EEDetId(hit->id());
      return geo->getPosition(EEid);
    } else {  // (hit->id().subdetId() == EcalBarrel)
      EBDetId EBid = EBDetId(hit->id());
      return geo->getPosition(EBid);
    }
  }

  double getRawEnergy(HBHERecHitCollection::const_iterator hit, int useRaw) {
    double energy = ((useRaw == 1) ? hit->eraw() : ((useRaw == 2) ? hit->eaux() : hit->energy()));
    return energy;
  }

  double getRawEnergy(EcalRecHitCollection::const_iterator hit, int) { return hit->energy(); }

  double getRawEnergy(edm::PCaloHitContainer::const_iterator hit, int) { return hit->energy(); }
}  // namespace spr
