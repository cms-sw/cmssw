#include <iostream>

#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"

const edm::ESHandle<CaloGeometry> *ClusterShapeAlgo::storedGeoHandle_ = NULL;
const EcalRecHitCollection  *ClusterShapeAlgo::storedRecHitsMap_ = NULL;

void ClusterShapeAlgo::Initialize(EcalRecHitCollection const *passedRecHitsMap,
				  const edm::ESHandle<CaloGeometry> *geoHandle)
{
  storedRecHitsMap_ = passedRecHitsMap;
  storedGeoHandle_ = geoHandle;
}

reco::ClusterShape ClusterShapeAlgo::Calculate(reco::BasicCluster passedCluster)
{
 if(storedRecHitsMap_ == NULL || storedGeoHandle_ == NULL)
    throw(std::runtime_error("\n\nOh No! ClusterShapeAlgo::Calculate called unitialized.\n\n"));
   
  ClusterShapeAlgo dataHolder;
  
  dataHolder.Calculate_TopEnergy(passedCluster);
  dataHolder.Calculate_2ndEnergy(passedCluster);
  dataHolder.Create_Map();
  dataHolder.Calculate_e2x2();
  dataHolder.Calculate_e3x2();
  dataHolder.Calculate_e3x3();
  dataHolder.Calculate_e5x5();
  dataHolder.Calculate_Location();
  dataHolder.Calculate_Covariances();
   
  return reco::ClusterShape( dataHolder.covEtaEta_, 
			     dataHolder.covEtaPhi_,dataHolder.covPhiPhi_, 
			     dataHolder.eMax_, dataHolder.eMaxId_, 
			     dataHolder.e2nd_, dataHolder.e2ndId_,
			     dataHolder.e2x2_, dataHolder.e3x2_, 
			     dataHolder.e3x3_, dataHolder.e5x5_,
			     dataHolder.e3x2Ratio_, dataHolder.location_);
}

void ClusterShapeAlgo::Calculate_TopEnergy(reco::BasicCluster passedCluster)
{
  double eMax=0;
  DetId eMaxId;

  std::vector<DetId> clusterDetIds = passedCluster.getHitsByDetId();
  std::vector<DetId>::iterator posCurrent;

  EcalRecHit testEcalRecHit;

  for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
  {
    if ((*posCurrent != DetId(0)) && (storedRecHitsMap_->find(*posCurrent) != storedRecHitsMap_->end()))
    {
      EcalRecHitCollection::const_iterator itt = storedRecHitsMap_->find(*posCurrent);
      testEcalRecHit = *itt;
      
      if(testEcalRecHit.energy() > eMax)
	{
	  eMax = testEcalRecHit.energy();
	  eMaxId = testEcalRecHit.id();
	} 
    }
  }
  
  eMax_ = eMax;
  eMaxId_ = eMaxId;
}

void ClusterShapeAlgo::Calculate_2ndEnergy(reco::BasicCluster passedCluster)
{
  double e2nd=0;
  DetId e2ndId;

  std::vector<DetId> clusterDetIds = passedCluster.getHitsByDetId();
  std::vector<DetId>::iterator posCurrent;

  EcalRecHit testEcalRecHit;

  for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
  { 
    if ((*posCurrent != DetId(0)) && (storedRecHitsMap_->find(*posCurrent) != storedRecHitsMap_->end()))
    {
      EcalRecHitCollection::const_iterator itt = storedRecHitsMap_->find(*posCurrent);
      testEcalRecHit = *itt;
      
      if(testEcalRecHit.energy() > e2nd && testEcalRecHit.id() != eMaxId_)
	{
	  e2nd = testEcalRecHit.energy();
	  e2ndId = testEcalRecHit.id();
	} 
    }
  }
 
  e2nd_ = e2nd;
  e2ndId_ = e2ndId;  
}

void ClusterShapeAlgo::Create_Map()
{
  EcalRecHit tempEcalRecHit;
  CaloNavigator<DetId> posCurrent;

  switch(eMaxId_.subdetId())
  {
     case EcalBarrel:    posCurrent = *(new CaloNavigator<DetId>(eMaxId_, new EcalBarrelTopology(*storedGeoHandle_))); break;
     case EcalEndcap:    posCurrent = *(new CaloNavigator<DetId>(eMaxId_, new EcalEndcapTopology(*storedGeoHandle_))); break;
     case EcalPreshower: posCurrent = *(new CaloNavigator<DetId>(eMaxId_, new EcalPreshowerTopology(*storedGeoHandle_))); break;
     default: throw(std::runtime_error("\n\nClusterShapeAlgo: No known topology for given subdetId. Giving up... =(\n\n"));
  }
  
  for(int x = 0; x < 5; x++)
    for(int y = 0; y < 5; y++)
    {
      posCurrent.home();
      posCurrent.offsetBy(-2+x,-2+y);

      if((*posCurrent != DetId(0)) && (storedRecHitsMap_->find(*posCurrent) != storedRecHitsMap_->end()))
      {
	EcalRecHitCollection::const_iterator itt = storedRecHitsMap_->find(*posCurrent);
	tempEcalRecHit = *itt;
	energyMap_[y][x] =  std::make_pair(tempEcalRecHit.id(),tempEcalRecHit.energy());
      }
      else
	energyMap_[y][x] = std::make_pair(DetId(0), 0);  
    }
  
  /*
  //Prints map for testing purposes, remove in final. 
  std::cout << "\n\n\n";

  for(int i = 0; i <= 4; i++)
  {
      std::cout << std::endl;
    for(int j = 0; j <= 4; j++)
    {
      std::cout.width(10);
      std::cout << std::left << energyMap_[i][j].second;
    }
  }

  std::cout << "\n\n\n" << std::endl;
  */
  
}

void ClusterShapeAlgo::Calculate_e2x2()
{
  double e2x2Max = 0;
  double e2x2Test = 0;

  int deltaX=0, deltaY=0;

  for(int corner = 0; corner < 4; corner++)
  {
    switch(corner)
    {
      case 0: deltaX = -1; deltaY = -1; break;
      case 1: deltaX = -1; deltaY =  1; break;
      case 2: deltaX =  1; deltaY = -1; break;
      case 3: deltaX =  1; deltaY =  1; break;
    }
    e2x2Test  = energyMap_[2][2].second;
    e2x2Test += energyMap_[2+deltaY][2].second;
    e2x2Test += energyMap_[2][2+deltaX].second;
    e2x2Test += energyMap_[2+deltaY][2+deltaX].second;

    e2x2Max = std::max(e2x2Test, e2x2Max);
  }

  e2x2_ = e2x2Max;

}

void ClusterShapeAlgo::Calculate_e3x2()
{
  double e3x2 = 0;
  double e3x2Ratio=0, e3x2RatioNumerator, e3x2RatioDenominator;

  int e2ndX = 2, e2ndY=2; 
  bool e2ndInX = false;

  int deltaY = 0, deltaX = 0;

  for(deltaX = -1; deltaX <= 1 && e2ndX == 2 && e2ndY == 2; deltaX++)
  {
    if(deltaX == 0)
      for(deltaY = -1; deltaY <= 1 && e2ndX == 2 && e2ndY == 2; deltaY+=2)
	{if(e2ndId_ == energyMap_[2+deltaY][2].first) e2ndY += deltaY; e2ndInX = false;}
    else 
     	{if(e2ndId_ == energyMap_[2][2+deltaX].first) e2ndX += deltaX; e2ndInX = true;} 
  }

  switch(e2ndInX)
  {
    case true:  deltaY = 1; deltaX = 0; break;
    case false: deltaY = 0; deltaX = 1; break; 
  }

  for(int sign = -1; sign <= 1; sign++)
      e3x2 += (energyMap_[2+deltaY*sign][2+deltaX*sign].second 
	      + std::max(0.0,energyMap_[e2ndY+deltaY*sign][e2ndX+deltaX*sign].second));
  
  e3x2RatioNumerator   = (std::max(0.0,energyMap_[e2ndY+deltaY][e2ndX+deltaX].second)
			  + std::max(0.0,energyMap_[e2ndY-deltaY][e2ndX-deltaX].second));
  e3x2RatioDenominator = (0.5 + energyMap_[2+deltaY][2+deltaX].second 
			  + energyMap_[2-deltaY][2-deltaX].second);
  e3x2Ratio = e3x2RatioNumerator /e3x2RatioDenominator;

  e3x2_ = e3x2;
  e3x2Ratio_ = e3x2Ratio;

} 

void ClusterShapeAlgo::Calculate_e3x3()
{
  double e3x3=0; 

  for(int i = 1; i <= 3; i++)
    for(int j = 1; j <= 3; j++)
      e3x3 += energyMap_[i][j].second;

  e3x3_ = e3x3;

}

void ClusterShapeAlgo::Calculate_e5x5()
{
  double e5x5=0; 

  for(int i = 0; i <= 4; i++)
    for(int j = 0; j <= 4; j++)
      e5x5 += energyMap_[i][j].second;

  e5x5_ = e5x5;

}

void ClusterShapeAlgo::Calculate_Location()
{
  std::vector<DetId> usedDetIds;

  for(int i = 0; i <= 4; i++)
    for(int j = 0; j <= 4; j++)
      if(!energyMap_[i][j].first.null()) usedDetIds.push_back(energyMap_[i][j].first);

  location_ = PositionCalc::Calculate_Location(usedDetIds);
}


void ClusterShapeAlgo::Calculate_Covariances()
{
  std::vector<DetId> usedDetIds;

  for(int i = 0; i <= 4; i++)
    for(int j = 0; j <= 4; j++)
      if(!energyMap_[i][j].first.null()) usedDetIds.push_back(energyMap_[i][j].first);

  std::map<std::string,double> covReturned = 
    PositionCalc::Calculate_Covariances(location_,usedDetIds);

  covEtaEta_ = covReturned.find("covEtaEta")->second;
  covEtaPhi_ = covReturned.find("covEtaPhi")->second;
  covPhiPhi_ = covReturned.find("covPhiPhi")->second;
}

