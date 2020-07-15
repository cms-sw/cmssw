
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"

#include "CondFormats/ESObjects/interface/ESEEIntercalibConstants.h"

#include <TMath.h>
#include <cmath>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

using namespace std;

PFEnergyCalibration::PFEnergyCalibration() {
  //calibChrisClean.C calibration parameters bhumika Nov, 2018
  faBarrel = std::make_unique<TF1>(
      "faBarrel", "[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))", 0., 1000.);
  faBarrel->SetParameter(0, -30.7141);
  faBarrel->SetParameter(1, 31.7583);
  faBarrel->SetParameter(2, 4.40594);
  faBarrel->SetParameter(3, 1.70914);
  faBarrel->SetParameter(4, 0.0613696);
  faBarrel->SetParameter(5, 0.000104857);
  faBarrel->SetParameter(6, -1.38927);
  faBarrel->SetParameter(7, -0.743082);
  fbBarrel = std::make_unique<TF1>(
      "fbBarrel", "[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))", 0., 1000.);
  fbBarrel->SetParameter(0, 2.25366);
  fbBarrel->SetParameter(1, 0.537715);
  fbBarrel->SetParameter(2, -4.81375);
  fbBarrel->SetParameter(3, 12.109);
  fbBarrel->SetParameter(4, 1.80577);
  fbBarrel->SetParameter(5, 0.187919);
  fbBarrel->SetParameter(6, -6.26234);
  fbBarrel->SetParameter(7, -0.607392);
  fcBarrel = std::make_unique<TF1>(
      "fcBarrel", "[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))", 0., 1000.);
  fcBarrel->SetParameter(0, 1.5125962);
  fcBarrel->SetParameter(1, 0.855057);
  fcBarrel->SetParameter(2, -6.04199);
  fcBarrel->SetParameter(3, 2.08229);
  fcBarrel->SetParameter(4, 0.592266);
  fcBarrel->SetParameter(5, 0.0291232);
  fcBarrel->SetParameter(6, 0.364802);
  fcBarrel->SetParameter(7, -1.50142);
  faEtaBarrelEH = std::make_unique<TF1>("faEtaBarrelEH", "[0]+[1]*exp(-x/[2])", 0., 1000.);
  faEtaBarrelEH->SetParameter(0, 0.0185555);
  faEtaBarrelEH->SetParameter(1, -0.0470674);
  faEtaBarrelEH->SetParameter(2, 396.959);
  fbEtaBarrelEH = std::make_unique<TF1>("fbEtaBarrelEH", "[0]+[1]*exp(-x/[2])", 0., 1000.);
  fbEtaBarrelEH->SetParameter(0, 0.0396458);
  fbEtaBarrelEH->SetParameter(1, 0.114128);
  fbEtaBarrelEH->SetParameter(2, 251.405);
  faEtaBarrelH = std::make_unique<TF1>("faEtaBarrelH", "[0]+[1]*x", 0., 1000.);
  faEtaBarrelH->SetParameter(0, 0.00434994);
  faEtaBarrelH->SetParameter(1, -5.16564e-06);
  fbEtaBarrelH = std::make_unique<TF1>("fbEtaBarrelH", "[0]+[1]*exp(-x/[2])", 0., 1000.);
  fbEtaBarrelH->SetParameter(0, -0.0232604);
  fbEtaBarrelH->SetParameter(1, 0.0937525);
  fbEtaBarrelH->SetParameter(2, 34.9935);

  faEndcap = std::make_unique<TF1>(
      "faEndcap", "[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))", 0., 1000.);
  faEndcap->SetParameter(0, 1.17227);
  faEndcap->SetParameter(1, 13.1489);
  faEndcap->SetParameter(2, -29.1672);
  faEndcap->SetParameter(3, 0.604223);
  faEndcap->SetParameter(4, 0.0426363);
  faEndcap->SetParameter(5, 3.30898e-15);
  faEndcap->SetParameter(6, 0.165293);
  faEndcap->SetParameter(7, -7.56786);
  fbEndcap = std::make_unique<TF1>(
      "fbEndcap", "[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))", 0., 1000.);
  fbEndcap->SetParameter(0, -0.974251);
  fbEndcap->SetParameter(1, 1.61733);
  fbEndcap->SetParameter(2, 0.0629183);
  fbEndcap->SetParameter(3, 7.78495);
  fbEndcap->SetParameter(4, -0.774289);
  fbEndcap->SetParameter(5, 7.81399e-05);
  fbEndcap->SetParameter(6, 0.139116);
  fbEndcap->SetParameter(7, -4.25551);
  fcEndcap = std::make_unique<TF1>(
      "fcEndcap", "[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))", 0., 1000.);
  fcEndcap->SetParameter(0, 1.01863);
  fcEndcap->SetParameter(1, 1.29787);
  fcEndcap->SetParameter(2, -3.97293);
  fcEndcap->SetParameter(3, 21.7805);
  fcEndcap->SetParameter(4, 0.810195);
  fcEndcap->SetParameter(5, 0.234134);
  fcEndcap->SetParameter(6, 1.42226);
  fcEndcap->SetParameter(7, -0.0997326);
  faEtaEndcapEH = std::make_unique<TF1>("faEtaEndcapEH", "[0]+[1]*exp(-x/[2])", 0., 1000.);
  faEtaEndcapEH->SetParameter(0, 0.0112692);
  faEtaEndcapEH->SetParameter(1, -2.68063);
  faEtaEndcapEH->SetParameter(2, 2.90973);
  fbEtaEndcapEH = std::make_unique<TF1>("fbEtaEndcapEH", "[0]+[1]*exp(-x/[2])", 0., 1000.);
  fbEtaEndcapEH->SetParameter(0, -0.0192991);
  fbEtaEndcapEH->SetParameter(1, -0.265);
  fbEtaEndcapEH->SetParameter(2, 80.5502);
  faEtaEndcapH = std::make_unique<TF1>("faEtaEndcapH", "[0]+[1]*exp(-x/[2])+[3]*[3]*exp(-x*x/([4]*[4]))", 0., 1000.);
  faEtaEndcapH->SetParameter(0, -0.0106029);
  faEtaEndcapH->SetParameter(1, -0.692207);
  faEtaEndcapH->SetParameter(2, 0.0542991);
  faEtaEndcapH->SetParameter(3, -0.171435);
  faEtaEndcapH->SetParameter(4, -61.2277);
  fbEtaEndcapH = std::make_unique<TF1>("fbEtaEndcapH", "[0]+[1]*exp(-x/[2])+[3]*[3]*exp(-x*x/([4]*[4]))", 0., 1000.);
  fbEtaEndcapH->SetParameter(0, 0.0214894);
  fbEtaEndcapH->SetParameter(1, -0.266704);
  fbEtaEndcapH->SetParameter(2, 5.2112);
  fbEtaEndcapH->SetParameter(3, 0.303578);
  fbEtaEndcapH->SetParameter(4, -104.367);

  //added by Bhumika on 2 august 2018

  fcEtaBarrelH = std::make_unique<TF1>("fcEtaBarrelH", "[3]*((x-[0])^[1])+[2]", 0., 1000.);
  fcEtaBarrelH->SetParameter(0, 0);
  fcEtaBarrelH->SetParameter(1, 2);
  fcEtaBarrelH->SetParameter(2, 0);
  fcEtaBarrelH->SetParameter(3, 1);

  fcEtaEndcapH = std::make_unique<TF1>("fcEtaEndcapH", "[3]*((x-[0])^[1])+[2]", 0., 1000.);
  fcEtaEndcapH->SetParameter(0, 0);
  fcEtaEndcapH->SetParameter(1, 0);
  fcEtaEndcapH->SetParameter(2, 0.05);
  fcEtaEndcapH->SetParameter(3, 0);

  fdEtaEndcapH = std::make_unique<TF1>("fdEtaEndcapH", "[3]*((x-[0])^[1])+[2]", 0., 1000.);
  fdEtaEndcapH->SetParameter(0, 1.5);
  fdEtaEndcapH->SetParameter(1, 4);
  fdEtaEndcapH->SetParameter(2, -1.1);
  fdEtaEndcapH->SetParameter(3, 1.0);

  fcEtaBarrelEH = std::make_unique<TF1>("fcEtaBarrelEH", "[3]*((x-[0])^[1])+[2]", 0., 1000.);
  fcEtaBarrelEH->SetParameter(0, 0);
  fcEtaBarrelEH->SetParameter(1, 2);
  fcEtaBarrelEH->SetParameter(2, 0);
  fcEtaBarrelEH->SetParameter(3, 1);

  fcEtaEndcapEH = std::make_unique<TF1>("fcEtaEndcapEH", "[3]*((x-[0])^[1])+[2]", 0., 1000.);
  fcEtaEndcapEH->SetParameter(0, 0);
  fcEtaEndcapEH->SetParameter(1, 0);
  fcEtaEndcapEH->SetParameter(2, 0);
  fcEtaEndcapEH->SetParameter(3, 0);

  fdEtaEndcapEH = std::make_unique<TF1>("fdEtaEndcapEH", "[3]*((x-[0])^[1])+[2]", 0., 1000.);
  fdEtaEndcapEH->SetParameter(0, 1.5);
  fdEtaEndcapEH->SetParameter(1, 2.0);
  fdEtaEndcapEH->SetParameter(2, 0.6);
  fdEtaEndcapEH->SetParameter(3, 1.0);
}

PFEnergyCalibration::CalibratedEndcapPFClusterEnergies PFEnergyCalibration::calibrateEndcapClusterEnergies(
    reco::PFCluster const& eeCluster,
    std::vector<reco::PFCluster const*> const& psClusterPointers,
    ESChannelStatus const& channelStatus,
    bool applyCrackCorrections) const {
  double ps1_energy_sum = 0.;
  double ps2_energy_sum = 0.;
  bool condP1 = true;
  bool condP2 = true;

  for (auto const& psclus : psClusterPointers) {
    bool cond = true;
    for (auto const& recH : psclus->recHitFractions()) {
      auto strip = recH.recHitRef()->detId();
      if (strip != ESDetId(0)) {
        //getStatusCode() == 0 => active channel
        // apply correction if all recHits are dead
        if (channelStatus.getMap().find(strip)->getStatusCode() == 0) {
          cond = false;
          break;
        }
      }
    }

    if (psclus->layer() == PFLayer::PS1) {
      ps1_energy_sum += psclus->energy();
      condP1 &= cond;
    } else if (psclus->layer() == PFLayer::PS2) {
      ps2_energy_sum += psclus->energy();
      condP2 &= cond;
    }
  }

  double ePS1 = condP1 ? -1. : 0.;
  double ePS2 = condP2 ? -1. : 0.;

  double cluscalibe = energyEm(eeCluster, ps1_energy_sum, ps2_energy_sum, ePS1, ePS2, applyCrackCorrections);

  return {cluscalibe, ePS1, ePS2};
}

void PFEnergyCalibration::energyEmHad(double t, double& e, double& h, double eta, double phi) const {
  // Use calorimetric energy as true energy for neutral particles
  const double tt = t;
  const double ee = e;
  const double hh = h;
  double etaCorrE = 1.;
  double etaCorrH = 1.;
  auto absEta = std::abs(eta);
  t = min(999.9, max(tt, e + h));
  if (t < 1.)
    return;

  // Barrel calibration
  if (absEta < 1.48) {
    // The energy correction
    double a = e > 0. ? aBarrel(t) : 1.;
    double b = e > 0. ? bBarrel(t) : cBarrel(t);
    double thresh = e > 0. ? threshE : threshH;

    // Protection against negative calibration
    if (a < -0.25 || b < -0.25) {
      a = 1.;
      b = 1.;
      thresh = 0.;
    }

    // The new estimate of the true energy
    t = min(999.9, max(tt, thresh + a * e + b * h));

    // The angular correction
    if (e > 0. && thresh > 0.) {
      etaCorrE = 1.0 + aEtaBarrelEH(t) + 1.3 * bEtaBarrelEH(t) * cEtaBarrelEH(absEta);
      etaCorrH = 1.0;
    } else {
      etaCorrE = 1.0 + aEtaBarrelH(t) + 1.3 * bEtaBarrelH(t) * cEtaBarrelH(absEta);
      etaCorrH = 1.0 + aEtaBarrelH(t) + bEtaBarrelH(t) * cEtaBarrelH(absEta);
    }
    if (e > 0. && thresh > 0.)
      e = h > 0. ? threshE - threshH + etaCorrE * a * e : threshE + etaCorrE * a * e;
    if (h > 0. && thresh > 0.) {
      h = threshH + etaCorrH * b * h;
    }

    // Endcap calibration
  } else {
    // The energy correction
    double a = e > 0. ? aEndcap(t) : 1.;
    double b = e > 0. ? bEndcap(t) : cEndcap(t);
    double thresh = e > 0. ? threshE : threshH;

    // Protection against negative calibration
    if (a < -0.25 || b < -0.25) {
      a = 1.;
      b = 1.;
      thresh = 0.;
    }

    // The new estimate of the true energy
    t = min(999.9, max(tt, thresh + a * e + b * h));

    // The angular correction
    const double dEta = std::abs(absEta - 1.5);
    const double etaPow = dEta * dEta * dEta * dEta;

    if (e > 0. && thresh > 0.) {
      if (absEta < 2.5) {
        etaCorrE = 1. + aEtaEndcapEH(t) + bEtaEndcapEH(t) * cEtaEndcapEH(absEta);
      } else {
        etaCorrE = 1. + aEtaEndcapEH(t) + 1.3 * bEtaEndcapEH(t) * dEtaEndcapEH(absEta);
      }

      etaCorrH = 1. + aEtaEndcapEH(t) + bEtaEndcapEH(t) * (0.04 + etaPow);
    } else {
      etaCorrE = 1.;
      if (absEta < 2.5) {
        etaCorrH = 1. + aEtaEndcapH(t) + bEtaEndcapH(t) * cEtaEndcapH(absEta);
      } else {
        etaCorrH = 1. + aEtaEndcapH(t) + bEtaEndcapH(t) * dEtaEndcapH(absEta);
      }
    }

    //t = min(999.9,max(tt, thresh + etaCorrE*a*e + etaCorrH*b*h));

    if (e > 0. && thresh > 0.)
      e = h > 0. ? threshE - threshH + etaCorrE * a * e : threshE + etaCorrE * a * e;
    if (h > 0. && thresh > 0.) {
      h = threshH + etaCorrH * b * h;
    }
  }

  // Protection
  if (e < 0. || h < 0.) {
    // Some protection against crazy calibration
    if (e < 0.)
      e = ee;
    if (h < 0.)
      h = hh;
  }

  // And that's it !
}

// The calibration functions
double PFEnergyCalibration::aBarrel(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfa_BARREL, point);

  } else {
    return faBarrel->Eval(x);
  }
}

double PFEnergyCalibration::bBarrel(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfb_BARREL, point);

  } else {
    return fbBarrel->Eval(x);
  }
}

double PFEnergyCalibration::cBarrel(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfc_BARREL, point);

  } else {
    return fcBarrel->Eval(x);
  }
}

double PFEnergyCalibration::aEtaBarrelEH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfaEta_BARRELEH, point);

  } else {
    return faEtaBarrelEH->Eval(x);
  }
}

double PFEnergyCalibration::bEtaBarrelEH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfbEta_BARRELEH, point);

  } else {
    return fbEtaBarrelEH->Eval(x);
  }
}

double PFEnergyCalibration::aEtaBarrelH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfaEta_BARRELH, point);

  } else {
    return faEtaBarrelH->Eval(x);
  }
}

double PFEnergyCalibration::bEtaBarrelH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfbEta_BARRELH, point);

  } else {
    return fbEtaBarrelH->Eval(x);
  }
}

double PFEnergyCalibration::aEndcap(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfa_ENDCAP, point);

  } else {
    return faEndcap->Eval(x);
  }
}

double PFEnergyCalibration::bEndcap(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfb_ENDCAP, point);

  } else {
    return fbEndcap->Eval(x);
  }
}

double PFEnergyCalibration::cEndcap(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfc_ENDCAP, point);

  } else {
    return fcEndcap->Eval(x);
  }
}

double PFEnergyCalibration::aEtaEndcapEH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfaEta_ENDCAPEH, point);

  } else {
    return faEtaEndcapEH->Eval(x);
  }
}

double PFEnergyCalibration::bEtaEndcapEH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfbEta_ENDCAPEH, point);

  } else {
    return fbEtaEndcapEH->Eval(x);
  }
}

double PFEnergyCalibration::aEtaEndcapH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfaEta_ENDCAPH, point);

  } else {
    return faEtaEndcapH->Eval(x);
  }
}

double PFEnergyCalibration::bEtaEndcapH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfbEta_ENDCAPH, point);
  } else {
    return fbEtaEndcapH->Eval(x);
  }
}

//added by Bhumika Kansal on 3 august 2018

double PFEnergyCalibration::cEtaBarrelH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfcEta_BARRELH, point);

  } else {
    return fcEtaBarrelH->Eval(x);
  }
}
double PFEnergyCalibration::cEtaEndcapH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfcEta_ENDCAPH, point);

  } else {
    return fcEtaEndcapH->Eval(x);
  }
}

double PFEnergyCalibration::dEtaEndcapH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfdEta_ENDCAPH, point);

  } else {
    return fdEtaEndcapH->Eval(x);
  }
}

double PFEnergyCalibration::cEtaBarrelEH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfcEta_BARRELEH, point);

  } else {
    return fcEtaBarrelEH->Eval(x);
  }
}

double PFEnergyCalibration::cEtaEndcapEH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfcEta_ENDCAPEH, point);

  } else {
    return fcEtaEndcapEH->Eval(x);
  }
}

double PFEnergyCalibration::dEtaEndcapEH(double x) const {
  if (pfCalibrations) {
    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfdEta_ENDCAPEH, point);

  } else {
    return fdEtaEndcapEH->Eval(x);
  }
}

double PFEnergyCalibration::energyEm(const reco::PFCluster& clusterEcal,
                                     double ePS1,
                                     double ePS2,
                                     bool crackCorrection) const {
  return Ecorr(clusterEcal.energy(), ePS1, ePS2, clusterEcal.eta(), clusterEcal.phi(), crackCorrection);
}

double PFEnergyCalibration::energyEm(const reco::PFCluster& clusterEcal,
                                     double ePS1,
                                     double ePS2,
                                     double& ps1,
                                     double& ps2,
                                     bool crackCorrection) const {
  return Ecorr(clusterEcal.energy(), ePS1, ePS2, clusterEcal.eta(), clusterEcal.phi(), ps1, ps2, crackCorrection);
}

std::ostream& operator<<(std::ostream& out, const PFEnergyCalibration& calib) {
  if (!out)
    return out;

  out << "PFEnergyCalibration -- " << endl;

  if (calib.pfCalibrations) {
    static const std::map<std::string, PerformanceResult::ResultType> functType = {
        {"PFfa_BARREL", PerformanceResult::PFfa_BARREL},
        {"PFfa_ENDCAP", PerformanceResult::PFfa_ENDCAP},
        {"PFfb_BARREL", PerformanceResult::PFfb_BARREL},
        {"PFfb_ENDCAP", PerformanceResult::PFfb_ENDCAP},
        {"PFfc_BARREL", PerformanceResult::PFfc_BARREL},
        {"PFfc_ENDCAP", PerformanceResult::PFfc_ENDCAP},
        {"PFfaEta_BARRELH", PerformanceResult::PFfaEta_BARRELH},
        {"PFfaEta_ENDCAPH", PerformanceResult::PFfaEta_ENDCAPH},
        {"PFfbEta_BARRELH", PerformanceResult::PFfbEta_BARRELH},
        {"PFfbEta_ENDCAPH", PerformanceResult::PFfbEta_ENDCAPH},
        {"PFfaEta_BARRELEH", PerformanceResult::PFfaEta_BARRELEH},
        {"PFfaEta_ENDCAPEH", PerformanceResult::PFfaEta_ENDCAPEH},
        {"PFfbEta_BARRELEH", PerformanceResult::PFfbEta_BARRELEH},
        {"PFfbEta_ENDCAPEH", PerformanceResult::PFfbEta_ENDCAPEH},
        {"PFfcEta_BARRELH", PerformanceResult::PFfcEta_BARRELH},
        {"PFfcEta_ENDCAPH", PerformanceResult::PFfcEta_ENDCAPH},
        {"PFfdEta_ENDCAPH", PerformanceResult::PFfdEta_ENDCAPH},
        {"PFfcEta_BARRELEH", PerformanceResult::PFfcEta_BARRELEH},
        {"PFfcEta_ENDCAPEH", PerformanceResult::PFfcEta_ENDCAPEH},
        {"PFfdEta_ENDCAPEH", PerformanceResult::PFfdEta_ENDCAPEH}

    };

    for (std::map<std::string, PerformanceResult::ResultType>::const_iterator func = functType.begin();
         func != functType.end();
         ++func) {
      cout << "Function: " << func->first << endl;
      PerformanceResult::ResultType fType = func->second;
      calib.pfCalibrations->printFormula(fType);
    }

  } else {
    std::cout << "Default calibration functions : " << std::endl;

    calib.faBarrel->Print();
    calib.fbBarrel->Print();
    calib.fcBarrel->Print();
    calib.faEtaBarrelEH->Print();
    calib.fbEtaBarrelEH->Print();
    calib.faEtaBarrelH->Print();
    calib.fbEtaBarrelH->Print();
    calib.faEndcap->Print();
    calib.fbEndcap->Print();
    calib.fcEndcap->Print();
    calib.faEtaEndcapEH->Print();
    calib.fbEtaEndcapEH->Print();
    calib.faEtaEndcapH->Print();
    calib.fbEtaEndcapH->Print();
    //
  }

  return out;
}

///////////////////////////////////////////////////////////////
////                                                       ////
////             CORRECTION OF PHOTONS' ENERGY             ////
////                                                       ////
////              Material effect: No tracker              ////
////       Tuned on CMSSW_2_1_0_pre4, Full Sim events      ////
////                                                       ////
///////////////////////////////////////////////////////////////
////                                                       ////
////            Jonathan Biteau - June 2008                ////
////                                                       ////
///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////
////                                                       ////
////  USEFUL FUNCTIONS FOR THE CORRECTION IN THE BARREL    ////
////                                                       ////
///////////////////////////////////////////////////////////////

//useful to compute the signed distance to the closest crack in the barrel
double PFEnergyCalibration::minimum(double a, double b) const {
  if (std::abs(b) < std::abs(a))
    a = b;
  return a;
}

namespace {
  constexpr double pi = M_PI;  // 3.14159265358979323846;

  std::vector<double> fillcPhi() {
    std::vector<double> retValue;
    retValue.resize(18, 0);
    retValue[0] = 2.97025;
    for (unsigned i = 1; i <= 17; ++i)
      retValue[i] = retValue[0] - 2 * i * pi / 18;

    return retValue;
  }

  //Location of the 18 phi-cracks
  const std::vector<double> cPhi = fillcPhi();
}  // namespace

//compute the unsigned distance to the closest phi-crack in the barrel
double PFEnergyCalibration::dCrackPhi(double phi, double eta) const {
  //Shift of this location if eta<0
  constexpr double delta_cPhi = 0.00638;

  double m;  //the result

  //the location is shifted
  if (eta < 0)
    phi += delta_cPhi;

  if (phi >= -pi && phi <= pi) {
    //the problem of the extrema
    if (phi < cPhi[17] || phi >= cPhi[0]) {
      if (phi < 0)
        phi += 2 * pi;
      m = minimum(phi - cPhi[0], phi - cPhi[17] - 2 * pi);
    }

    //between these extrema...
    else {
      bool OK = false;
      unsigned i = 16;
      while (!OK) {
        if (phi < cPhi[i]) {
          m = minimum(phi - cPhi[i + 1], phi - cPhi[i]);
          OK = true;
        } else
          i -= 1;
      }
    }
  } else {
    m = 0.;  //if there is a problem, we assum that we are in a crack
    std::cout << "Problem in dminphi" << std::endl;
  }
  if (eta < 0)
    m = -m;  //because of the disymetry
  return m;
}

// corrects the effect of phi-cracks
double PFEnergyCalibration::CorrPhi(double phi, double eta) const {
  // we use 3 gaussians to correct the phi-cracks effect
  constexpr double p1 = 5.59379e-01;
  constexpr double p2 = -1.26607e-03;
  constexpr double p3 = 9.61133e-04;

  constexpr double p4 = 1.81691e-01;
  constexpr double p5 = -4.97535e-03;
  constexpr double p6 = 1.31006e-03;

  constexpr double p7 = 1.38498e-01;
  constexpr double p8 = 1.18599e-04;
  constexpr double p9 = 2.01858e-03;

  double dminphi = dCrackPhi(phi, eta);

  double result =
      (1 + p1 * TMath::Gaus(dminphi, p2, p3) + p4 * TMath::Gaus(dminphi, p5, p6) + p7 * TMath::Gaus(dminphi, p8, p9));

  return result;
}

// corrects the effect of  |eta|-cracks
double PFEnergyCalibration::CorrEta(double eta) const {
  // we use a gaussian with a screwness for each of the 5 |eta|-cracks
  constexpr double a[] = {6.13349e-01, 5.08146e-01, 4.44480e-01, 3.3487e-01, 7.65627e-01};     // amplitude
  constexpr double m[] = {-1.79514e-02, 4.44747e-01, 7.92824e-01, 1.14090e+00, 1.47464e+00};   // mean
  constexpr double s[] = {7.92382e-03, 3.06028e-03, 3.36139e-03, 3.94521e-03, 8.63950e-04};    // sigma
  constexpr double sa[] = {1.27228e+01, 3.81517e-02, 1.63507e-01, -6.56480e-02, 1.87160e-01};  // screwness amplitude
  constexpr double ss[] = {5.48753e-02, -1.00223e-02, 2.22866e-03, 4.26288e-04, 2.67937e-03};  // screwness sigma
  double result = 1;

  for (unsigned i = 0; i <= 4; i++)
    result += a[i] * TMath::Gaus(eta, m[i], s[i]) *
              (1 + sa[i] * TMath::Sign(1., eta - m[i]) * TMath::Exp(-std::abs(eta - m[i]) / ss[i]));

  return result;
}

//corrects the global behaviour in the barrel
double PFEnergyCalibration::CorrBarrel(double E, double eta) const {
  //Energy dependency
  /*
  //YM Parameters 52XX:
  constexpr double p0=1.00000e+00;
  constexpr double p1=3.27753e+01;
  constexpr double p2=2.28552e-02;
  constexpr double p3=3.06139e+00;
  constexpr double p4=2.25135e-01;
  constexpr double p5=1.47824e+00;
  constexpr double p6=1.09e-02;
  constexpr double p7=4.19343e+01;
  */
  constexpr double p0 = 0.9944;
  constexpr double p1 = 9.827;
  constexpr double p2 = 1.503;
  constexpr double p3 = 1.196;
  constexpr double p4 = 0.3349;
  constexpr double p5 = 0.89;
  constexpr double p6 = 0.004361;
  constexpr double p7 = 51.51;
  //Eta dependency
  constexpr double p8 = 2.705593e-03;

  double result =
      (p0 + 1 / (p1 + p2 * TMath::Power(E, p3)) + p4 * TMath::Exp(-E / p5) + p6 * TMath::Exp(-E * E / (p7 * p7))) *
      (1 + p8 * eta * eta);

  return result;
}

///////////////////////////////////////////////////////////////
////                                                       ////
////  USEFUL FUNCTIONS FOR THE CORRECTION IN THE ENDCAPS   ////
////  Parameters tuned for:                                ////
////          dR(ClustersPS1,ClusterEcal) < 0.08           ////
////          dR(ClustersPS2,ClusterEcal) < 0.13           ////
////                                                       ////
///////////////////////////////////////////////////////////////

//Alpha, Beta, Gamma give the weight of each sub-detector (PS layer1, PS layer2 and Ecal) in the areas of the endcaps where there is a PS
// Etot = Beta*eEcal + Gamma*(ePS1 + Alpha*ePS2)

double PFEnergyCalibration::Alpha(double eta) const {
  //Energy dependency
  constexpr double p0 = 5.97621e-01;

  //Eta dependency
  constexpr double p1 = -1.86407e-01;
  constexpr double p2 = 3.85197e-01;

  //so that <feta()> = 1
  constexpr double norm = (p1 + p2 * (2.6 + 1.656) / 2);

  double result = p0 * (p1 + p2 * eta) / norm;

  return result;
}

double PFEnergyCalibration::Beta(double E, double eta) const {
  //Energy dependency
  constexpr double p0 = 0.032;
  constexpr double p1 = 9.70394e-02;
  constexpr double p2 = 2.23072e+01;
  constexpr double p3 = 100;

  //Eta dependency
  constexpr double p4 = 1.02496e+00;
  constexpr double p5 = -4.40176e-03;

  //so that <feta()> = 1
  constexpr double norm = (p4 + p5 * (2.6 + 1.656) / 2);

  double result = (1.0012 + p0 * TMath::Exp(-E / p3) + p1 * TMath::Exp(-E / p2)) * (p4 + p5 * eta) / norm;
  return result;
}

double PFEnergyCalibration::Gamma(double etaEcal) const {
  //Energy dependency
  constexpr double p0 = 2.49752e-02;

  //Eta dependency
  constexpr double p1 = 6.48816e-02;
  constexpr double p2 = -1.59517e-02;

  //so that <feta()> = 1
  constexpr double norm = (p1 + p2 * (2.6 + 1.656) / 2);

  double result = p0 * (p1 + p2 * etaEcal) / norm;

  return result;
}

///////////////////////////////////////////////////////////////
////                                                       ////
////   THE CORRECTIONS IN THE BARREL AND IN THE ENDCAPS    ////
////                                                       ////
///////////////////////////////////////////////////////////////

// returns the corrected energy in the barrel (0,1.48)
// Global Behaviour, phi and eta cracks are taken into account
double PFEnergyCalibration::EcorrBarrel(double E, double eta, double phi, bool crackCorrection) const {
  // double result = E*CorrBarrel(E,eta)*CorrEta(eta)*CorrPhi(phi,eta);
  double correction = crackCorrection ? std::max(CorrEta(eta), CorrPhi(phi, eta)) : 1.;
  double result = E * CorrBarrel(E, eta) * correction;

  return result;
}

// returns the corrected energy in the area between the barrel and the PS (1.48,1.65)
double PFEnergyCalibration::EcorrZoneBeforePS(double E, double eta) const {
  //Energy dependency
  constexpr double p0 = 1;
  constexpr double p1 = 0.18;
  constexpr double p2 = 8.;

  //Eta dependency
  constexpr double p3 = 0.3;
  constexpr double p4 = 1.11;
  constexpr double p5 = 0.025;
  constexpr double p6 = 1.49;
  constexpr double p7 = 0.6;

  //so that <feta()> = 1
  constexpr double norm = 1.21;

  double result = E * (p0 + p1 * TMath::Exp(-E / p2)) * (p3 + p4 * TMath::Gaus(eta, p6, p5) + p7 * eta) / norm;

  return result;
}

// returns the corrected energy in the PS (1.65,2.6)
// only when (ePS1>0)||(ePS2>0)
double PFEnergyCalibration::EcorrPS(double eEcal, double ePS1, double ePS2, double etaEcal) const {
  // gives the good weights to each subdetector
  double E = Beta(1.0155 * eEcal + 0.025 * (ePS1 + 0.5976 * ePS2) / 9e-5, etaEcal) * eEcal +
             Gamma(etaEcal) * (ePS1 + Alpha(etaEcal) * ePS2) / 9e-5;

  //Correction of the residual energy dependency
  constexpr double p0 = 1.00;
  constexpr double p1 = 2.18;
  constexpr double p2 = 1.94;
  constexpr double p3 = 4.13;
  constexpr double p4 = 1.127;

  double result = E * (p0 + p1 * TMath::Exp(-E / p2) - p3 * TMath::Exp(-E / p4));

  return result;
}

// returns the corrected energy in the PS (1.65,2.6)
// only when (ePS1>0)||(ePS2>0)
double PFEnergyCalibration::EcorrPS(
    double eEcal, double ePS1, double ePS2, double etaEcal, double& outputPS1, double& outputPS2) const {
  // gives the good weights to each subdetector
  double gammaprime = Gamma(etaEcal) / 9e-5;

  if (outputPS1 == 0 && outputPS2 == 0 && esEEInterCalib_ != nullptr) {
    // both ES planes working
    // scaling factor accounting for data-mc
    outputPS1 = gammaprime * ePS1 * esEEInterCalib_->getGammaLow0();
    outputPS2 = gammaprime * Alpha(etaEcal) * ePS2 * esEEInterCalib_->getGammaLow3();
  } else if (outputPS1 == 0 && outputPS2 == -1 && esEEInterCalib_ != nullptr) {
    // ESP1 only working
    double corrTotES = gammaprime * ePS1 * esEEInterCalib_->getGammaLow0() * esEEInterCalib_->getGammaLow1();
    outputPS1 = gammaprime * ePS1 * esEEInterCalib_->getGammaLow0();
    outputPS2 = corrTotES - outputPS1;
  } else if (outputPS1 == -1 && outputPS2 == 0 && esEEInterCalib_ != nullptr) {
    // ESP2 only working
    double corrTotES =
        gammaprime * Alpha(etaEcal) * ePS2 * esEEInterCalib_->getGammaLow3() * esEEInterCalib_->getGammaLow2();
    outputPS2 = gammaprime * Alpha(etaEcal) * ePS2 * esEEInterCalib_->getGammaLow3();
    outputPS1 = corrTotES - outputPS2;
  } else {
    // none working
    outputPS1 = gammaprime * ePS1;
    outputPS2 = gammaprime * Alpha(etaEcal) * ePS2;
  }

  double E = Beta(1.0155 * eEcal + 0.025 * (ePS1 + 0.5976 * ePS2) / 9e-5, etaEcal) * eEcal + outputPS1 + outputPS2;

  //Correction of the residual energy dependency
  constexpr double p0 = 1.00;
  constexpr double p1 = 2.18;
  constexpr double p2 = 1.94;
  constexpr double p3 = 4.13;
  constexpr double p4 = 1.127;

  double corrfac = (p0 + p1 * TMath::Exp(-E / p2) - p3 * TMath::Exp(-E / p4));
  outputPS1 *= corrfac;
  outputPS2 *= corrfac;
  double result = E * corrfac;

  return result;
}

// returns the corrected energy in the PS (1.65,2.6)
// only when (ePS1=0)&&(ePS2=0)
double PFEnergyCalibration::EcorrPS_ePSNil(double eEcal, double eta) const {
  //Energy dependency
  constexpr double p0 = 1.02;
  constexpr double p1 = 0.165;
  constexpr double p2 = 6.5;
  constexpr double p3 = 2.1;

  //Eta dependency
  constexpr double p4 = 1.02496e+00;
  constexpr double p5 = -4.40176e-03;

  //so that <feta()> = 1
  constexpr double norm = (p4 + p5 * (2.6 + 1.656) / 2);

  double result = eEcal * (p0 + p1 * TMath::Exp(-std::abs(eEcal - p3) / p2)) * (p4 + p5 * eta) / norm;

  return result;
}

// returns the corrected energy in the area between the end of the PS and the end of the endcap (2.6,2.98)
double PFEnergyCalibration::EcorrZoneAfterPS(double E, double eta) const {
  //Energy dependency
  constexpr double p0 = 1;
  constexpr double p1 = 0.058;
  constexpr double p2 = 12.5;
  constexpr double p3 = -1.05444e+00;
  constexpr double p4 = -5.39557e+00;
  constexpr double p5 = 8.38444e+00;
  constexpr double p6 = 6.10998e-01;

  //Eta dependency
  constexpr double p7 = 1.06161e+00;
  constexpr double p8 = 0.41;
  constexpr double p9 = 2.918;
  constexpr double p10 = 0.0181;
  constexpr double p11 = 2.05;
  constexpr double p12 = 2.99;
  constexpr double p13 = 0.0287;

  //so that <feta()> = 1
  constexpr double norm = 1.045;

  double result = E * (p0 + p1 * TMath::Exp(-(E - p3) / p2) + 1 / (p4 + p5 * TMath::Power(E, p6))) *
                  (p7 + p8 * TMath::Gaus(eta, p9, p10) + p11 * TMath::Gaus(eta, p12, p13)) / norm;
  return result;
}

// returns the corrected energy everywhere
// this work should be improved between 1.479 and 1.52 (junction barrel-endcap)
double PFEnergyCalibration::Ecorr(
    double eEcal, double ePS1, double ePS2, double eta, double phi, bool crackCorrection) const {
  constexpr double endBarrel = 1.48;
  constexpr double beginingPS = 1.65;
  constexpr double endPS = 2.6;
  constexpr double endEndCap = 2.98;

  double result = 0;

  eta = std::abs(eta);

  if (eEcal > 0) {
    if (eta <= endBarrel)
      result = EcorrBarrel(eEcal, eta, phi, crackCorrection);
    else if (eta <= beginingPS)
      result = EcorrZoneBeforePS(eEcal, eta);
    else if ((eta < endPS) && ePS1 == 0 && ePS2 == 0)
      result = EcorrPS_ePSNil(eEcal, eta);
    else if (eta < endPS)
      result = EcorrPS(eEcal, ePS1, ePS2, eta);
    else if (eta < endEndCap)
      result = EcorrZoneAfterPS(eEcal, eta);
    else
      result = eEcal;
  } else
    result = eEcal;  // useful if eEcal=0 or eta>2.98
  //protection
  if (result < eEcal)
    result = eEcal;
  return result;
}

// returns the corrected energy everywhere
// this work should be improved between 1.479 and 1.52 (junction barrel-endcap)
double PFEnergyCalibration::Ecorr(double eEcal,
                                  double ePS1,
                                  double ePS2,
                                  double eta,
                                  double phi,
                                  double& ps1,
                                  double& ps2,
                                  bool crackCorrection) const {
  constexpr double endBarrel = 1.48;
  constexpr double beginingPS = 1.65;
  constexpr double endPS = 2.6;
  constexpr double endEndCap = 2.98;

  double result = 0;

  eta = std::abs(eta);

  if (eEcal > 0) {
    if (eta <= endBarrel)
      result = EcorrBarrel(eEcal, eta, phi, crackCorrection);
    else if (eta <= beginingPS)
      result = EcorrZoneBeforePS(eEcal, eta);
    else if ((eta < endPS) && ePS1 == 0 && ePS2 == 0)
      result = EcorrPS_ePSNil(eEcal, eta);
    else if (eta < endPS)
      result = EcorrPS(eEcal, ePS1, ePS2, eta, ps1, ps2);
    else if (eta < endEndCap)
      result = EcorrZoneAfterPS(eEcal, eta);
    else
      result = eEcal;
  } else
    result = eEcal;  // useful if eEcal=0 or eta>2.98
  // protection
  if (result < eEcal)
    result = eEcal;
  return result;
}
