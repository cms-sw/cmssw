
#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

l1t::CaloTower::CaloTower( const LorentzVector& p4,
		   double etEm,
		   double etHad,
		   int pt,
		   int eta,
		   int phi,
		   int qual,
		   int hwEtEm,
		   int hwEtHad)
  : L1Candidate(p4, pt, eta, phi, qual),
    etEm_(etEm),
    etHad_(etHad),
    hwEtEm_(hwEtEm),
    hwEtHad_(hwEtHad)
{
  
}

l1t::CaloTower::~CaloTower() 
{

}

void l1t::CaloTower::setEtEm(double et)
{
  etEm_ = et;
}

void l1t::CaloTower::setEtHad(double et)
{
  etHad_ = et;
}

void l1t::CaloTower::setHwEtEm(int et)
{
  hwEtEm_ = et;
}

void l1t::CaloTower::setHwEtHad(int et)
{
  hwEtHad_ = et;
}


double l1t::CaloTower::etEm()
{
  return etEm_;
}

double l1t::CaloTower::etHad()
{
  return etHad_;
}

int l1t::CaloTower::hwEtEm()
{
  return hwEtEm_;
}

int l1t::CaloTower::hwEtHad()
{
  return hwEtHad_;
}
