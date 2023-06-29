#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

HcalHitRelabeller::HcalHitRelabeller(bool nd) : theRecNumber(nullptr), neutralDensity_(nd) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HcalHitRelabeller initialized with"
                              << " neutralDensity " << neutralDensity_;
#endif
}

void HcalHitRelabeller::process(std::vector<PCaloHit>& hcalHits) {
  if (theRecNumber) {
#ifdef EDM_ML_DEBUG
    int ii(0);
#endif
    for (auto& hcalHit : hcalHits) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalSim") << "Hit[" << ii << "] " << std::hex << hcalHit.id() << std::dec << " Neutral density "
                                  << neutralDensity_;
#endif
      uint32_t hid;
      int det, z, depth, eta, phi, layer;
      HcalTestNumbering::unpackHcalIndex(hcalHit.id(), det, z, depth, eta, phi, layer);
      if ((det == 2) && (layer == 2) && (eta == 18))
	depth = 2;
      hid = HcalTestNumbering::packHcalIndex(det, z, depth, eta, phi, layer);
      double wt = (neutralDensity_) ? (energyWt(hid)) : 1.0;
      double energy = wt * (hcalHit.energy());
      hcalHit.setEnergy(energy);
      uint32_t newid = relabel(hid).rawId();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalSim") << "Hit " << ii << " out of " << hcalHits.size() << " " << std::hex << newid
                                  << std::dec << " E " << energy << " wt " << wt;
#endif
      hcalHit.setID(newid);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalSim") << "Final Setting::subdet: " << HcalDetId(newid).subdet() << " z: " << HcalDetId(newid).zside() << " depth: " << HcalDetId(newid).depth() << " ieta: " << HcalDetId(newid).ietaAbs() << " iphi: " << HcalDetId(newid).iphi() << " wt " << wt;
      ++ii;
#endif
    }
  } else {
    edm::LogWarning("HcalSim") << "HcalHitRelabeller: no valid HcalDDDRecConstants";
  }
}

void HcalHitRelabeller::setGeometry(const HcalDDDRecConstants*& recNum) { theRecNumber = recNum; }

DetId HcalHitRelabeller::relabel(const uint32_t testId) const {
  return HcalHitRelabeller::relabel(testId, theRecNumber);
}

DetId HcalHitRelabeller::relabel(const uint32_t testId, const HcalDDDRecConstants* theRecNumber) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "Enter HcalHitRelabeller::relabel";
#endif
  HcalDetId hid;
  int det, z, depth, eta, phi, layer, sign;
  HcalTestNumbering::unpackHcalIndex(testId, det, z, depth, eta, phi, layer);

  sign = (z == 0) ? (-1) : (1);
  HcalDDDRecConstants::HcalID id = theRecNumber->getHCID(det, sign * eta, phi, layer, depth);
  int depth0 = ((det == 1) && (layer == 2) && (eta == 15)) ? depth : id.depth;

  if (id.subdet == int(HcalBarrel)) {
    hid = HcalDetId(HcalBarrel, sign * id.eta, id.phi, depth0);
  } else if (id.subdet == int(HcalEndcap)) {
    hid = HcalDetId(HcalEndcap, sign * id.eta, id.phi, depth0);
  } else if (id.subdet == int(HcalOuter)) {
    hid = HcalDetId(HcalOuter, sign * id.eta, id.phi, depth0);
  } else if (id.subdet == int(HcalForward)) {
    hid = HcalDetId(HcalForward, sign * id.eta, id.phi, depth0);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "Initial indices:" << "subdet: " << det << " " << "z: " << z << " "   << "depth: " << depth << " " << "ieta: " << eta << " "  << "iphi: " << phi << " " << "layer: " << layer << " new HcalDetId -> hex.RawID = " << std::hex << hid.rawId() << std::dec << " subdet, z, depth, eta, phi = " << det << " " << z << " " << id.depth << " " << id.eta << " " << id.phi << " ---> " << hid;
#endif
  return hid;
}

double HcalHitRelabeller::energyWt(const uint32_t testId) const {
  int det, z, depth, eta, phi, layer;
  HcalTestNumbering::unpackHcalIndex(testId, det, z, depth, eta, phi, layer);
  int zside = (z == 0) ? (-1) : (1);
  double wt = ((((det == 1) && (layer <= 1)) || ((det == 2) && (layer <= 2))) && (depth == 1))
                  ? theRecNumber->getLayer0Wt(det, phi, zside)
                  : 1.0;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "EnergyWT::det: " << det << " z: " << z << ":" << zside << " depth: " << depth
                              << " ieta: " << eta << " iphi: " << phi << " layer: " << layer << " wt " << wt;
#endif
  return wt;
}
