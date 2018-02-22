///
/// \class HGCalTowerMap2DImpl
///
/// \author: Thomas Strebler
///
/// Description: first iteration of HGCal Tower Maps


#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTowerMap3DImpl.h"


HGCalTowerMap3DImpl::HGCalTowerMap3DImpl( )
{

}





void HGCalTowerMap3DImpl::buildTowerMap3D(const std::vector<edm::Ptr<l1t::HGCalTowerMap>> & towerMapsPtrs,
					  l1t::HGCalTowerBxCollection & towers
				       ){

  l1t::HGCalTowerMap towerMap;

  for( std::vector<edm::Ptr<l1t::HGCalTowerMap>>::const_iterator map = towerMapsPtrs.begin(); map != towerMapsPtrs.end(); ++map ){
    if(towerMap.nEtaBins()==0) towerMap = (**map);
    else towerMap += (**map);
  }

  int nEtaBins = towerMap.nEtaBins();
  int nPhiBins = towerMap.nPhiBins();

  vector<l1t::HGCalTower> towersTmp;

  for(int iEta=-nEtaBins; iEta<=nEtaBins; iEta++){
    if(iEta==0) continue;
    for(int iPhi=0; iPhi<nPhiBins; iPhi++){ 
      l1t::HGCalTower tower = towerMap.tower(iEta,iPhi);
      if(tower.pt()>0) towersTmp.push_back(tower);
    }
  }

  towers.resize(0, towersTmp.size());
  int i=0;
  for(auto tower : towersTmp){
    towers.set( 0, i, tower);
    i++;
  }


}
