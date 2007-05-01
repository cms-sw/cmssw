#include "RecoEgamma/EgammaMCTools/interface/PizeroMCTruth.h"

#include <iostream>

PizeroMCTruth::PizeroMCTruth() {

}

void PizeroMCTruth::SetDecay(float r, float z, CLHEP::HepLorentzVector momentum1, CLHEP::HepLorentzVector momentum2)
{
  dalitz_ = false;
  r_ = r;
  z_ = z;
  momentum1_ = momentum1;
  momentum2_ = momentum2;
}

void PizeroMCTruth::SetDalitzDecay(float r, float z, CLHEP::HepLorentzVector momentum1, CLHEP::HepLorentzVector momentum2, CLHEP::HepLorentzVector momentum3)
{
  dalitz_ = true;
  r_ = r;
  z_ = z;
  momentum1_ = momentum1;
  momentum2_ = momentum2;
  momentum3_ = momentum3;
}
