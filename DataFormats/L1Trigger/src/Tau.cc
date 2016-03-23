
#include "DataFormats/L1Trigger/interface/Tau.h"
using namespace l1t;

void Tau::clear_extended(){
  towerIEta_ = 0;
  towerIPhi_ = 0;
  rawEt_ = 0;
  isoEt_ = 0;
  nTT_ = 0;
  hasEM_ = false;
  isMerged_ = false;
}

Tau::Tau( const LorentzVector& p4,
	       int pt,
	       int eta,
	       int phi,
	       int qual,
	       int iso )
  : L1Candidate(p4, pt, eta, phi, qual,iso)
{
  clear_extended();
}

Tau::Tau( const PolarLorentzVector& p4,
	       int pt,
	       int eta,
	       int phi,
	       int qual,
	       int iso )
  : L1Candidate(p4, pt, eta, phi, qual,iso)
{
  clear_extended();
}

Tau::~Tau()
{

}

void Tau::setTowerIEta(short int ieta) {
  towerIEta_ = ieta;
}

void Tau::setTowerIPhi(short int iphi) {
  towerIPhi_ = iphi;
}

void Tau::setRawEt(short int et) { 
  rawEt_ = et;
}

void Tau::setIsoEt(short int et) {
  isoEt_ = et;
}

void Tau::setNTT(short int ntt) {
  nTT_ = ntt;
}

void Tau::setHasEM(bool hasEM) {
  hasEM_ = hasEM;
}

void Tau::setIsMerged(bool isMerged) {
  isMerged_ = isMerged;
}

short int Tau::towerIEta() const {
  return towerIEta_;
}

short int Tau::towerIPhi() const {
  return towerIPhi_;
}

short int Tau::rawEt() const {
  return rawEt_;
}

short int Tau::isoEt() const {
  return isoEt_;
}

short int Tau::nTT() const {
  return nTT_;
}

bool Tau::hasEM() const {
  return hasEM_;
}

bool Tau::isMerged() const {
  return isMerged_;
}

