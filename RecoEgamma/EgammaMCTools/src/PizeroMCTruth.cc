#include "RecoEgamma/EgammaMCTools/interface/PizeroMCTruth.h"

#include <iostream>

PizeroMCTruth::PizeroMCTruth() {

}

void PizeroMCTruth::SetDecay(float r, float z, CLHEP::HepLorentzVector momentum1, CLHEP::HepLorentzVector momentum2)
{
  r_ = r;
  z_ = z;
  momentum1_ = momentum1;
  momentum2_ = momentum2;
}
