#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "DataFormats/Math/interface/normalizedPhi.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace l1t;

constexpr double HGCalTowerMap::kEtaMax_;
constexpr double HGCalTowerMap::kEtaMin_;
constexpr double HGCalTowerMap::kEtaMaxLoose_;
constexpr double HGCalTowerMap::kEtaMinLoose_;
constexpr double HGCalTowerMap::kPhiMax_;
constexpr double HGCalTowerMap::kPhiMin_;


HGCalTowerMap::HGCalTowerMap( int nEtaBins, int nPhiBins )
{

  nEtaBins_ = nEtaBins;
  nPhiBins_ = nPhiBins;

  double eta_step = (kEtaMax_ - kEtaMin_)/nEtaBins_;
  for(int i=0; i<nEtaBins_; i++) etaBins_.emplace_back( kEtaMin_ + eta_step*i );
  etaBins_.emplace_back( kEtaMax_ );
  
  double phi_step = (kPhiMax_ - kPhiMin_)/nPhiBins_;
  for(int i=0; i<nPhiBins_; i++) phiBins_.emplace_back( kPhiMin_ + phi_step*i );
  phiBins_.emplace_back( kPhiMax_ );


  for(int iEta=-nEtaBins_; iEta<=nEtaBins_; iEta++){
    if(iEta==0) continue;
    for(int iPhi=0; iPhi<nPhiBins_; iPhi++){
      l1t::HGCalTower tower;
      tower.setHwEta(iEta);
      tower.setHwPhi(iPhi);
      int bin = bin_id(iEta,iPhi);
      towerMap_[bin] = tower;
    }
  }


}


HGCalTowerMap::HGCalTowerMap( const vector<double>& etaBins, const vector<double>& phiBins )
{

  nEtaBins_ = etaBins.size()-1;
  nPhiBins_ = phiBins.size()-1;
  etaBins_ = etaBins;
  phiBins_ = phiBins;

  for(int iEta=-nEtaBins_; iEta<=nEtaBins_; iEta++){
    if(iEta==0) continue;
    for(int iPhi=0; iPhi<nPhiBins_; iPhi++){
      l1t::HGCalTower tower;
      tower.setHwEta(iEta);
      tower.setHwPhi(iPhi);
      int bin = bin_id(iEta,iPhi);
      towerMap_[bin] = tower;
    }
  }

}


HGCalTowerMap::~HGCalTowerMap()
{

}


int HGCalTowerMap::iEta(const double eta) const
{

  double absEta = fabs(eta);
  int signEta = eta>0 ? 1 : -1;
  
  if(absEta<kEtaMin_){
    absEta = kEtaMin_;
    if(absEta<kEtaMinLoose_) edm::LogWarning("HGCalTowerMap") << "WARNING: trying to access towers below min eta range of tower maps eta="<<eta;
    return signEta;
  }
  else if(absEta>kEtaMax_){
    absEta = kEtaMax_;
    if(absEta>kEtaMaxLoose_) edm::LogWarning("HGCalTowerMap") << "WARNING: trying to access towers above max eta range of tower maps eta="<<eta;
    return signEta*nEtaBins_;
  }

  for(int i = 0; i<nEtaBins_; i++){
    if(absEta>=etaBins_[i] && absEta<etaBins_[i+1]){      
      return signEta*(i+1);
    }
  }
  
  return 0;

}


int HGCalTowerMap::iPhi(const double phi) const
{

  double phi_norm = normalizedPhi(phi);
 
  for(int i = 0; i<nPhiBins_; i++){
    if(phi_norm>=phiBins_[i] && phi_norm<phiBins_[i+1]){
      return i;
    }
  }
  
  return -1;

}


HGCalTowerMap& HGCalTowerMap::operator+=(const HGCalTowerMap& map){

  if(etaBins_!=map.etaBins() || phiBins_!=map.phiBins()){
    throw edm::Exception(edm::errors::StdException, "StdException")
      << "HGCalTowerMap: Trying to add HGCalTowerMaps with different bins"<<endl;
  }

  for(int iEta=-nEtaBins_; iEta<=nEtaBins_; iEta++){
    if(iEta==0) continue;
    for(int iPhi=0; iPhi<nPhiBins_; iPhi++){
      towerMap_[bin_id(iEta,iPhi)] += map.tower(iEta,iPhi);
    }
  }

  return *this;

}



int HGCalTowerMap::bin_id(int iEta, int iPhi) const{

  if(iEta==0 || iEta>nEtaBins_ || iEta<-nEtaBins_){
    throw edm::Exception(edm::errors::StdException, "StdException")
      << "HGCalTowerMap: Trying to access a bin out of eta range"<<endl;
  }

  if(iPhi<0 || iPhi>=nPhiBins_){
    throw edm::Exception(edm::errors::StdException, "StdException")
      << "HGCalTowerMap: Trying to access a bin out of phi range"<<endl;
  }

  return iEta*nPhiBins_ + iPhi;

}

