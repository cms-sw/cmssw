#include "DataFormats/L1Trigger/interface/Jet.h"
using namespace l1t;

void Jet::clear_extended(){
  towerIEta_ = 0;
  towerIPhi_ = 0;
  rawEt_ = 0;
  seedEt_ = 0;
  puEt_ = 0;
  puDonutEt_[0] = puDonutEt_[1] = puDonutEt_[2] = puDonutEt_[3] = 0;
}

Jet::Jet( const LorentzVector& p4,
	       int pt,
	       int eta,
	       int phi,
	       int qual )
  : L1Candidate(p4, pt, eta, phi, qual, 0)
{
  clear_extended();
}

Jet::Jet( const PolarLorentzVector& p4,
	       int pt,
	       int eta,
	       int phi,
	       int qual )
  : L1Candidate(p4, pt, eta, phi, qual, 0)
{
  clear_extended();
}

Jet::~Jet()
{

}

void Jet::setTowerIEta(short int ieta) {
  towerIEta_ = ieta;
}

void Jet::setTowerIPhi(short int iphi) {
  towerIPhi_ = iphi;
}

void Jet::setSeedEt(short int et) {
  seedEt_ = et;
}

void Jet::setRawEt(short int et) {
  rawEt_ = et;
}

void Jet::setPUEt(short int et) {
  puEt_ = et;
}

void Jet::setPUDonutEt(int i, short int et) {
  if (i>=0 && i<4) puDonutEt_[i] = et;
}

short int Jet::towerIEta() {
  return towerIEta_;
}

short int Jet::towerIPhi() {
  return towerIPhi_;
}

short int Jet::seedEt() {
  return seedEt_;
}

short int Jet::rawEt() {
  return rawEt_;
}

short int Jet::puEt() {
  return puEt_;
}

short int Jet::puDonutEt(int i) {
  if (i>=0 && i<4) return puDonutEt_[i];
  else return 0;
}
