//*****************************************************************************
// File:      EgammaHcalIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************
//C++ includes
#include <vector>
#include <functional>

//ROOT includes
#include <Math/VectorUtil.h>

//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

using namespace std;

double scaleToE(const double& eta) { return 1.0; }
double scaleToEt(const double& eta) { return sin(2 * atan(exp(-eta))); }

EgammaHcalIsolation::EgammaHcalIsolation(double extRadius,
                                         double intRadius,
                                         double eLowB,
                                         double eLowE,
                                         double etLowB,
                                         double etLowE,
                                         edm::ESHandle<CaloGeometry> theCaloGeom,
                                         const HBHERecHitCollection& mhbhe)
    : extRadius_(extRadius),
      intRadius_(intRadius),
      eLowB_(eLowB),
      eLowE_(eLowE),
      etLowB_(etLowB),
      etLowE_(etLowE),
      theCaloGeom_(theCaloGeom),
      mhbhe_(mhbhe) {
  //set up the geometry and selector
  const CaloGeometry* caloGeom = theCaloGeom_.product();
  doubleConeSel_ = new CaloDualConeSelector<HBHERecHit>(intRadius_, extRadius_, caloGeom, DetId::Hcal);
}

EgammaHcalIsolation::~EgammaHcalIsolation() { delete doubleConeSel_; }

double EgammaHcalIsolation::getHcalSum(const GlobalPoint& pclu,
                                       const HcalDepth& depth,
                                       double (*scale)(const double&)) const {
  double sum = 0.;
  if (!mhbhe_.empty()) {
    //Compute the HCAL energy behind ECAL
    doubleConeSel_->selectCallback(pclu, mhbhe_, [this, &sum, &depth, &scale](const HBHERecHit& i) {
      double eta = theCaloGeom_.product()->getPosition(i.detid()).eta();
      HcalDetId hcalDetId(i.detid());
      if (hcalDetId.subdet() == HcalBarrel &&         //Is it in the barrel?
          i.energy() > eLowB_ &&                      //Does it pass the min energy?
          i.energy() * scaleToEt(eta) > etLowB_ &&    //Does it pass the min et?
          (depth == AllDepths || depth == Depth1)) {  //Are we asking for the first depth?
        sum += i.energy() * scale(eta);
      }
      if (hcalDetId.subdet() == HcalEndcap &&       //Is it in the endcap?
          i.energy() > eLowE_ &&                    //Does it pass the min energy?
          i.energy() * scaleToEt(eta) > etLowE_) {  //Does it pass the min et?
        switch (depth) {                            //Which depth?
          case AllDepths:
            sum += i.energy() * scale(eta);
            break;
          case Depth1:
            sum += (isDepth2(i.detid())) ? 0 : i.energy() * scale(eta);
            break;
          case Depth2:
            sum += (isDepth2(i.detid())) ? i.energy() * scale(eta) : 0;
            break;
        }
      }
    });
  }

  return sum;
}

bool EgammaHcalIsolation::isDepth2(const DetId& detId) const {
  if ((HcalDetId(detId).depth() == 2 && HcalDetId(detId).ietaAbs() >= 18 && HcalDetId(detId).ietaAbs() < 27) ||
      (HcalDetId(detId).depth() == 3 && HcalDetId(detId).ietaAbs() == 27)) {
    return true;

  } else {
    return false;
  }
}
