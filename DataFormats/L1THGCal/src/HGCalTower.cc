#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "FWCore/Utilities/interface/EDMException.h"

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




HGCalTower& HGCalTower::operator+=(const HGCalTower tower){

  if(this->hwEta()!= tower.hwEta() || this->hwPhi()!= tower.hwPhi()){
    throw edm::Exception(edm::errors::StdException, "StdException")
      << "HGCalTower: Trying to add HGCalTowers with different coordinates"<<endl;
  }

  this->setP4(this->p4() + tower.p4());
  this->setEtEm(this->etEm() + tower.etEm());
  this->setEtHad(this->etHad() + tower.etHad());

  this->setHwPt(this->hwPt() + tower.hwPt());
  this->setHwEtEm(this->hwEtEm() + tower.hwEtEm());
  this->setHwEtHad(this->hwEtHad() + tower.hwEtHad());

  return *this;

}
