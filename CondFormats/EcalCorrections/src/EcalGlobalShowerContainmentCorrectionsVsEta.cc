// Implementation of class EcalGlobalShowerContainmentCorrectionsVsEta
// Author: Paolo Meridiani
// $Id: EcalGlobalShowerContainmentCorrectionsVsEta.cc,v 1.1 2007/07/13 17:37:04 meridian Exp $

#include "CondFormats/EcalCorrections/interface/EcalGlobalShowerContainmentCorrectionsVsEta.h"
#include <DataFormats/DetId/interface/DetId.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
//#include <iostream>

const EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients
EcalGlobalShowerContainmentCorrectionsVsEta::correctionCoefficients() const {
  return coefficients_;
}

void EcalGlobalShowerContainmentCorrectionsVsEta::fillCorrectionCoefficients(const Coefficients& coefficients) {
  coefficients_ = coefficients;
}

/** Calculate the correction for the given direction and type  */
const double EcalGlobalShowerContainmentCorrectionsVsEta::correction(
    const DetId& xtal, EcalGlobalShowerContainmentCorrectionsVsEta::Type type) const {
  if (xtal.det() == DetId::Ecal && xtal.subdetId() == EcalBarrel) {
    double corr = 0;

    if (EBDetId(xtal).ieta() < coefficients_.data[0])
      corr = coefficients_.data[1];
    else
      corr = coefficients_.data[1] + coefficients_.data[2] * pow(EBDetId(xtal).ieta() - coefficients_.data[0], 2);

    return corr;
  } else if (xtal.det() == DetId::Ecal && xtal.subdetId() == EcalEndcap)
    return 1.;
  else
    return -1;
}

const double EcalGlobalShowerContainmentCorrectionsVsEta::correction3x3(const DetId& xtal) const {
  double corr = correction(xtal, e3x3);
  return corr;
}

const double EcalGlobalShowerContainmentCorrectionsVsEta::correction5x5(const DetId& xtal) const {
  double corr = correction(xtal, e5x5);
  return corr;
}
