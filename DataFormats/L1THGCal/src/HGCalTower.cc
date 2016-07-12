#include "DataFormats/L1THGCal/interface/HGCalTower.h"

using namespace l1t;

HGCalTower::HGCalTower( const LorentzVector& p4,
                       double etEm,
                       double etHad,
                       int pt,
                       int eta,
                       int phi,
                       int qual,
                       int hwEtEm,
                       int hwEtHad,
                       int hwEtRatio)
  : L1Candidate(p4, pt, eta, phi, qual),
    etEm_(etEm),
    etHad_(etHad),
    hwEtEm_(hwEtEm),
    hwEtHad_(hwEtHad),
    hwEtRatio_(hwEtRatio)
{
  
}

HGCalTower::~HGCalTower() 
{

}

void HGCalTower::setEtEm(double et)
{
  etEm_ = et;
}

void HGCalTower::setEtHad(double et)
{
  etHad_ = et;
}

void HGCalTower::setHwEtEm(int et)
{
  hwEtEm_ = et;
}

void HGCalTower::setHwEtHad(int et)
{
  hwEtHad_ = et;
}

void HGCalTower::setHwEtRatio(int ratio)
{
  hwEtRatio_ = ratio;
}

double HGCalTower::etEm()const
{
  return etEm_;
}

double HGCalTower::etHad()const
{
  return etHad_;
}

int HGCalTower::hwEtEm()const
{
  return hwEtEm_;
}

int HGCalTower::hwEtHad()const
{
  return hwEtHad_;
}

int HGCalTower::hwEtRatio()const
{
  return hwEtRatio_;
}
