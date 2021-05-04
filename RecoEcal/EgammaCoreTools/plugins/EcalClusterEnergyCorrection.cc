/** \class EcalClusterEnergyCorrection
  *  Function that provides supercluster energy correction due to Bremsstrahlung loss
  *
  *  $Id: EcalClusterEnergyCorrection.h
  *  $Date:
  *  $Revision:
  *  \author Yurii Maravin, KSU, March 2009
  */

#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionParametersRcd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionParameters.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"

class EcalClusterEnergyCorrection : public EcalClusterFunctionBaseClass {
public:
  EcalClusterEnergyCorrection(const edm::ParameterSet &){};

  // get/set explicit methods for parameters
  const EcalClusterEnergyCorrectionParameters *getParameters() const { return params_; }
  // check initialization
  void checkInit() const;

  // compute the correction
  float getValue(const reco::SuperCluster &, const int mode) const override;
  float getValue(const reco::BasicCluster &, const EcalRecHitCollection &) const override { return 0.; };

  // set parameters
  void init(const edm::EventSetup &es) override;

private:
  float fEta(float e, float eta, int algorithm) const;
  float fBrem(float e, float eta, int algorithm) const;
  float fEtEta(float et, float eta, int algorithm) const;

  edm::ESHandle<EcalClusterEnergyCorrectionParameters> esParams_;
  const EcalClusterEnergyCorrectionParameters *params_;
};

// Shower leakage corrections developed by Jungzhie et al. using TB data
// Developed for EB only!
float EcalClusterEnergyCorrection::fEta(float energy, float eta, int algorithm) const {
  // this correction is setup only for EB
  if (algorithm != 0)
    return energy;

  float ieta = fabs(eta) * (5 / 0.087);
  float p0 = (params_->params())[0];  // should be 40.2198
  float p1 = (params_->params())[1];  // should be -3.03103e-6

  float correctedEnergy = energy;
  if (ieta < p0)
    correctedEnergy = energy;
  else
    correctedEnergy = energy / (1.0 + p1 * (ieta - p0) * (ieta - p0));
  //std::cout << "ECEC fEta = " << correctedEnergy << std::endl;
  return correctedEnergy;
}

float EcalClusterEnergyCorrection::fBrem(float e, float brem, int algorithm) const {
  // brem == phiWidth/etaWidth of the SuperCluster
  // e  == energy of the SuperCluster
  // first parabola (for br > threshold)
  // p0 + p1*x + p2*x^2
  // second parabola (for br <= threshold)
  // ax^2 + bx + c, make y and y' the same in threshold
  // y = p0 + p1*threshold + p2*threshold^2
  // yprime = p1 + 2*p2*threshold
  // a = p3
  // b = yprime - 2*a*threshold
  // c = y - a*threshold^2 - b*threshold

  int offset;
  if (algorithm == 0)
    offset = 0;
  else if (algorithm == 1)
    offset = 20;
  else {
    // not supported, produce no correction
    return e;
  }

  //Make No Corrections if brem is invalid!
  if (brem == 0)
    return e;

  float bremLowThr = (params_->params())[2 + offset];
  float bremHighThr = (params_->params())[3 + offset];
  if (brem < bremLowThr)
    brem = bremLowThr;
  if (brem > bremHighThr)
    brem = bremHighThr;

  // Parameters provided in cfg file
  float p0 = (params_->params())[4 + offset];
  float p1 = (params_->params())[5 + offset];
  float p2 = (params_->params())[6 + offset];
  float p3 = (params_->params())[7 + offset];
  float p4 = (params_->params())[8 + offset];
  //
  float threshold = p4;

  float y = p0 * threshold * threshold + p1 * threshold + p2;
  float yprime = 2 * p0 * threshold + p1;
  float a = p3;
  float b = yprime - 2 * a * threshold;
  float c = y - a * threshold * threshold - b * threshold;

  float fCorr = 1;
  if (brem < threshold)
    fCorr = p0 * brem * brem + p1 * brem + p2;
  else
    fCorr = a * brem * brem + b * brem + c;

  //std::cout << "ECEC fBrem " << e/fCorr << std::endl;
  return e / fCorr;
}

float EcalClusterEnergyCorrection::fEtEta(float et, float eta, int algorithm) const {
  // et -- Et of the SuperCluster (with respect to (0,0,0))
  // eta -- eta of the SuperCluster

  //std::cout << "fEtEta, mode = " << algorithm << std::endl;
  //std::cout << "ECEC: p0    " << (params_->params())[9]  << " " << (params_->params())[10] << " " << (params_->params())[11] << " " << (params_->params())[12] << std::endl;
  //std::cout << "ECEC: p1    " << (params_->params())[13] << " " << (params_->params())[14] << " " << (params_->params())[15] << " " << (params_->params())[16] << std::endl;
  //std::cout << "ECEC: fcorr " << (params_->params())[17] << " " << (params_->params())[18] << " " << (params_->params())[19] << std::endl;

  float fCorr = 0.;
  int offset;
  if (algorithm == 0)
    offset = 0;
  else if (algorithm == 1)
    offset = 20;
  else if (algorithm == 10)
    offset = 28;
  else if (algorithm == 11)
    offset = 39;
  else {
    // not supported, produce no correction
    return et;
  }

  // Barrel
  if (algorithm == 0 || algorithm == 10) {
    float p0 = (params_->params())[9 + offset] +
               (params_->params())[10 + offset] / (et + (params_->params())[11 + offset]) +
               (params_->params())[12 + offset] / (et * et);
    float p1 = (params_->params())[13 + offset] +
               (params_->params())[14 + offset] / (et + (params_->params())[15 + offset]) +
               (params_->params())[16 + offset] / (et * et);

    fCorr = p0 + p1 * atan((params_->params())[17 + offset] * ((params_->params())[18 + offset] - fabs(eta))) +
            (params_->params())[19 + offset] * fabs(eta);

  } else if (algorithm == 1 || algorithm == 11) {  // Endcap
    float p0 = (params_->params())[9 + offset] + (params_->params())[10 + offset] / sqrt(et);
    float p1 = (params_->params())[11 + offset] + (params_->params())[12 + offset] / sqrt(et);
    float p2 = (params_->params())[13 + offset] + (params_->params())[14 + offset] / sqrt(et);
    float p3 = (params_->params())[15 + offset] + (params_->params())[16 + offset] / sqrt(et);

    fCorr = p0 + p1 * fabs(eta) + p2 * eta * eta + p3 / fabs(eta);
  }

  // cap the correction at 50%
  if (fCorr < 0.5)
    fCorr = 0.5;
  if (fCorr > 1.5)
    fCorr = 1.5;

  //std::cout << "ECEC fEtEta " << et/fCorr << std::endl;
  return et / fCorr;
}

float EcalClusterEnergyCorrection::getValue(const reco::SuperCluster &superCluster, const int mode) const {
  // mode flags:
  // hybrid modes: 1 - return f(eta) correction in GeV
  //               2 - return f(eta) + f(brem) correction
  //               3 - return f(eta) + f(brem) + f(et, eta) correction
  // multi5x5:     4 - return f(brem) correction
  //               5 - return f(brem) + f(et, eta) correction

  // special cases: mode = 10 -- return f(et, eta) correction with respect to already corrected SC in barrel
  //                mode = 11 -- return f(et, eta) correction with respect to already corrected SC in endcap

  checkInit();

  float eta = fabs(superCluster.eta());
  float brem = superCluster.phiWidth() / superCluster.etaWidth();
  int algorithm = -1;

  if (mode <= 3 || mode == 10) {
    // algorithm: hybrid
    algorithm = 0;

    float energy = superCluster.rawEnergy();
    if (mode == 10) {
      algorithm = 10;
      energy = superCluster.energy();
    }
    float correctedEnergy = fEta(energy, eta, algorithm);

    if (mode == 1) {
      return correctedEnergy - energy;

    } else {
      // now apply F(brem)
      correctedEnergy = fBrem(correctedEnergy, brem, algorithm);
      if (mode == 2) {
        return correctedEnergy - energy;
      }

      float correctedEt = correctedEnergy / cosh(eta);
      correctedEt = fEtEta(correctedEt, eta, algorithm);
      correctedEnergy = correctedEt * cosh(eta);
      return correctedEnergy - energy;
    }
  } else if (mode == 4 || mode == 5 || mode == 11) {
    algorithm = 1;

    float energy = superCluster.rawEnergy() + superCluster.preshowerEnergy();
    if (mode == 11) {
      algorithm = 11;
      energy = superCluster.energy();
    }

    float correctedEnergy = fBrem(energy, brem, algorithm);
    if (mode == 4) {
      return correctedEnergy - energy;
    }

    float correctedEt = correctedEnergy / cosh(eta);
    correctedEt = fEtEta(correctedEt, eta, algorithm);
    correctedEnergy = correctedEt * cosh(eta);
    return correctedEnergy - energy;

  } else {
    // perform no correction
    return 0;
  }
}

void EcalClusterEnergyCorrection::init(const edm::EventSetup &es) {
  es.get<EcalClusterEnergyCorrectionParametersRcd>().get(esParams_);
  params_ = esParams_.product();
}

void EcalClusterEnergyCorrection::checkInit() const {
  if (!params_) {
    // non-initialized function parameters: throw exception
    throw cms::Exception("EcalClusterEnergyCorrection::checkInit()")
        << "Trying to access an uninitialized crack correction function.\n"
           "Please call `init( edm::EventSetup &)' before any use of the function.\n";
  }
}

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
DEFINE_EDM_PLUGIN(EcalClusterFunctionFactory, EcalClusterEnergyCorrection, "EcalClusterEnergyCorrection");
