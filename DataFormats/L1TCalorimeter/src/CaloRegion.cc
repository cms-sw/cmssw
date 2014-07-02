
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

l1t::CaloRegion::CaloRegion( const LorentzVector& p4,
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

l1t::CaloRegion::~CaloRegion() 
{

}

void l1t::CaloRegion::setEtEm(double et)
{
  etEm_ = et;
}

void l1t::CaloRegion::setEtHad(double et)
{
  etHad_ = et;
}

void l1t::CaloRegion::setHwEtEm(int et)
{
  hwEtEm_ = et;
}

void l1t::CaloRegion::setHwEtHad(int et)
{
  hwEtHad_ = et;
}


double l1t::CaloRegion::etEm()const
{
  return etEm_;
}

double l1t::CaloRegion::etHad()const
{
  return etHad_;
}

int l1t::CaloRegion::hwEtEm()const
{
  return hwEtEm_;
}

int l1t::CaloRegion::hwEtHad()const
{
  return hwEtHad_;
}
