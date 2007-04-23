#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"

#include <iostream>

ElectronMCTruth::ElectronMCTruth() {

}

void ElectronMCTruth::SetBrem(float r, float z, float phoFrac, float eGamma, float eElectron) {

  r_=r;
  z_=z;
  phoFrac_=phoFrac;
  eGamma_=eGamma;
  eElectron_=eElectron;

}

float ElectronMCTruth::GetBremR() {
  return r_;
}

float ElectronMCTruth::GetBremZ() {
  return z_;
}

float ElectronMCTruth::GetBremFraction() {
  return phoFrac_;
}

float ElectronMCTruth::GetBremPhotonE() {
  return eGamma_;
}

float ElectronMCTruth::GetBremElectronE() {
  return eElectron_;
}


