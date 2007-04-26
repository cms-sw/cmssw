#include <iostream>

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "Geometry/EcalBarrelAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"

ClusterShapeAlgo::ClusterShapeAlgo(const PositionCalc& passedPositionCalc) {
  posCalculator_ = passedPositionCalc;
}

reco::ClusterShape ClusterShapeAlgo::Calculate(const reco::BasicCluster &passedCluster,
                                               const EcalRecHitCollection *hits,
                                               const CaloSubdetectorGeometry * geometry,
                                               const CaloSubdetectorTopology* topology)
{
  Calculate_TopEnergy(passedCluster,hits);
  Calculate_2ndEnergy(passedCluster,hits);
  Create_Map(hits,topology);
  Calculate_e2x2();
  Calculate_e3x2();
  Calculate_e3x3();
  Calculate_e4x4();
  Calculate_e5x5();
  Calculate_e2x5Right();
  Calculate_e2x5Left();
  Calculate_e2x5Top();
  Calculate_e2x5Bottom();
  Calculate_Covariances(passedCluster,hits,geometry);
  Calculate_BarrelBasketEnergyFraction(passedCluster,hits, Eta, geometry);
  Calculate_BarrelBasketEnergyFraction(passedCluster,hits, Phi, geometry);

  return reco::ClusterShape(covEtaEta_, covEtaPhi_, covPhiPhi_, eMax_, eMaxId_,
			    e2nd_, e2ndId_, e2x2_, e3x2_, e3x3_,e4x4_, e5x5_,
			    e2x5Right_, e2x5Left_, e2x5Top_, e2x5Bottom_,
			    e3x2Ratio_, energyBasketFractionEta_, energyBasketFractionPhi_);
}

void ClusterShapeAlgo::Calculate_TopEnergy(const reco::BasicCluster &passedCluster,const EcalRecHitCollection *hits)
{
  double eMax=0;
  DetId eMaxId(0);

  std::vector<DetId> clusterDetIds = passedCluster.getHitsByDetId();
  std::vector<DetId>::iterator posCurrent;

  EcalRecHit testEcalRecHit;

  for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
  {
    if ((*posCurrent != DetId(0)) && (hits->find(*posCurrent) != hits->end()))
    {
      EcalRecHitCollection::const_iterator itt = hits->find(*posCurrent);
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

void ClusterShapeAlgo::Calculate_2ndEnergy(const reco::BasicCluster &passedCluster,const EcalRecHitCollection *hits)
{
  double e2nd=0;
  DetId e2ndId(0);

  std::vector<DetId> clusterDetIds = passedCluster.getHitsByDetId();
  std::vector<DetId>::iterator posCurrent;

  EcalRecHit testEcalRecHit;

  for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
  { 
    if ((*posCurrent != DetId(0)) && (hits->find(*posCurrent) != hits->end()))
    {
      EcalRecHitCollection::const_iterator itt = hits->find(*posCurrent);
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

void ClusterShapeAlgo::Create_Map(const EcalRecHitCollection *hits,const CaloSubdetectorTopology* topology)
{
  EcalRecHit tempEcalRecHit;
  CaloNavigator<DetId> posCurrent = CaloNavigator<DetId>(eMaxId_,topology );

  for(int x = 0; x < 5; x++)
    for(int y = 0; y < 5; y++)
    {
      posCurrent.home();
      posCurrent.offsetBy(-2+x,-2+y);

      if((*posCurrent != DetId(0)) && (hits->find(*posCurrent) != hits->end()))
      {
				EcalRecHitCollection::const_iterator itt = hits->find(*posCurrent);
				tempEcalRecHit = *itt;
				energyMap_[y][x] = std::make_pair(tempEcalRecHit.id(),tempEcalRecHit.energy()); 
      }
      else
				energyMap_[y][x] = std::make_pair(DetId(0), 0);  
    }
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

    if(e2x2Test > e2x2Max)
    {
			e2x2Max = e2x2Test;
			e2x2_Diagonal_X_ = 2+deltaX;
			e2x2_Diagonal_Y_ = 2+deltaY;
    }
  }

  e2x2_ = e2x2Max;

}

void ClusterShapeAlgo::Calculate_e3x2()
{
  double e3x2 = 0.0;
  double e3x2Ratio = 0.0, e3x2RatioNumerator = 0.0, e3x2RatioDenominator = 0.0;

  int e2ndX = 2, e2ndY = 2;
  int deltaY = 0, deltaX = 0;

  double nextEnergy = -999;
  int nextEneryDirection = -1;

  for(int cardinalDirection = 0; cardinalDirection < 4; cardinalDirection++)
  {
    switch(cardinalDirection)
    {
      case 0: deltaX = -1; deltaY =  0; break;
      case 1: deltaX =  1; deltaY =  0; break;
      case 2: deltaX =  0; deltaY = -1; break;
      case 3: deltaX =  0; deltaY =  1; break;
    }
   
    if(energyMap_[2+deltaY][2+deltaX].second >= nextEnergy)
    {
        nextEnergy = energyMap_[2+deltaY][2+deltaX].second;
        nextEneryDirection = cardinalDirection;
       
        e2ndX = 2+deltaX;
        e2ndY = 2+deltaY;
    }
  }
 
  switch(nextEneryDirection)
  {
    case 0: ;
    case 1: deltaX = 0; deltaY = 1; break;
    case 2: ;
    case 3: deltaX = 1; deltaY = 0; break;
  }

  for(int sign = -1; sign <= 1; sign++)
      e3x2 += (energyMap_[2+deltaY*sign][2+deltaX*sign].second + energyMap_[e2ndY+deltaY*sign][e2ndX+deltaX*sign].second);
 
  e3x2RatioNumerator   = energyMap_[e2ndY+deltaY][e2ndX+deltaX].second + energyMap_[e2ndY-deltaY][e2ndX-deltaX].second;
  e3x2RatioDenominator = 0.5 + energyMap_[2+deltaY][2+deltaX].second + energyMap_[2-deltaY][2-deltaX].second;
  e3x2Ratio = e3x2RatioNumerator / e3x2RatioDenominator;

  e3x2_ = e3x2;
  e3x2Ratio_ = e3x2Ratio;
}  

void ClusterShapeAlgo::Calculate_e3x3()
{
  double e3x3=0;

  for(int i = 1; i <= 3; i++)
    for(int j = 1; j <= 3; j++)
      e3x3 += energyMap_[j][i].second;

  e3x3_ = e3x3;

}

void ClusterShapeAlgo::Calculate_e4x4()
{
  double e4x4=0;
	
  int startX=-1, startY=-1;

	switch(e2x2_Diagonal_X_)
	{
		case 1: startX = 0; break;
		case 3: startX = 1; break;
	}

	switch(e2x2_Diagonal_Y_)
	{
		case 1: startY = 0; break;
		case 3: startY = 1; break;
	}

  for(int i = startX; i <= startX+3; i++)
 	  for(int j = startY; j <= startY+3; j++)
   	  e4x4 += energyMap_[j][i].second;

  e4x4_ = e4x4;
}

void ClusterShapeAlgo::Calculate_e5x5()
{
  double e5x5=0;

  for(int i = 0; i <= 4; i++)
    for(int j = 0; j <= 4; j++)
      e5x5 += energyMap_[j][i].second;

  e5x5_ = e5x5;

}

void ClusterShapeAlgo::Calculate_e2x5Right()
{
double e2x5R=0.0;
  for(int i = 0; i <= 4; i++){
    for(int j = 0; j <= 4; j++){
      if(j>2){e2x5R +=energyMap_[i][j].second;}
    }
  }
  e2x5Right_=e2x5R;
}

void ClusterShapeAlgo::Calculate_e2x5Left()
{
double e2x5L=0.0;
  for(int i = 0; i <= 4; i++){
    for(int j = 0; j <= 4; j++){
      if(j<2){e2x5L +=energyMap_[i][j].second;}
    }
  }
  e2x5Left_=e2x5L;
}

void ClusterShapeAlgo::Calculate_e2x5Bottom()
{
double e2x5B=0.0;
  for(int i = 0; i <= 4; i++){
    for(int j = 0; j <= 4; j++){

      if(i>2){e2x5B +=energyMap_[i][j].second;}
    }
  }
  e2x5Bottom_=e2x5B;
}

void ClusterShapeAlgo::Calculate_e2x5Top()
{
double e2x5T=0.0;
  for(int i = 0; i <= 4; i++){
    for(int j = 0; j <= 4; j++){

      if(i<2){e2x5T +=energyMap_[i][j].second;}
    }
  }
  e2x5Top_=e2x5T;
}

void ClusterShapeAlgo::Calculate_Covariances(const reco::BasicCluster &passedCluster, const EcalRecHitCollection* hits, const CaloSubdetectorGeometry* geometry)
{
  std::vector<DetId> usedDetIds;
  covEtaEta_ = 0.;
  covEtaPhi_ = 0.;
  covPhiPhi_ = 0.;

  for(int i = 0; i <= 4; i++)
    for(int j = 0; j <= 4; j++)
      if(!energyMap_[i][j].first.null()) usedDetIds.push_back(energyMap_[i][j].first);

  std::map<std::string,double> covReturned = posCalculator_.Calculate_Covariances(passedCluster.position(),usedDetIds,hits,geometry);

  covEtaEta_ = covReturned.find("covEtaEta")->second;
  covEtaPhi_ = covReturned.find("covEtaPhi")->second;
  covPhiPhi_ = covReturned.find("covPhiPhi")->second;
}

void ClusterShapeAlgo::Calculate_BarrelBasketEnergyFraction(const reco::BasicCluster &passedCluster,
                                                            const EcalRecHitCollection *hits,
                                                            const int EtaPhi,
                                                            const CaloSubdetectorGeometry* geometry) 
{
  if(  (hits!=0) && ( ((*hits)[0]).id().subdetId() != EcalBarrel )  ) {
     //std::cout << "No basket correction for endacap!" << std::endl;
     return;
  }

  std::map<int,double> indexedBasketEnergy;
  std::vector<DetId> clusterDetIds = passedCluster.getHitsByDetId();
  const EcalBarrelGeometry* subDetGeometry = (const EcalBarrelGeometry*) geometry;

  for(std::vector<DetId>::iterator posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
  {
    int basketIndex = 999;

    if(EtaPhi == Eta) {
      int unsignedIEta = abs(EBDetId(*posCurrent).ieta());
      std::vector<int> etaBasketSize = subDetGeometry->getEtaBaskets();

      for(unsigned int i = 0; i < etaBasketSize.size(); i++) {
        unsignedIEta -= etaBasketSize[i];
        if(unsignedIEta - 1 < 0)
        {
          basketIndex = i;
          break;
        }
      }
      basketIndex = (basketIndex+1)*(EBDetId(*posCurrent).ieta() > 0 ? 1 : -1);

    } else if(EtaPhi == Phi) {
      int halfNumBasketInPhi = (EBDetId::MAX_IPHI - EBDetId::MIN_IPHI + 1)/subDetGeometry->getBasketSizeInPhi()/2;

      basketIndex = (EBDetId(*posCurrent).iphi() - 1)/subDetGeometry->getBasketSizeInPhi()
                  - (EBDetId(clusterDetIds[0]).iphi() - 1)/subDetGeometry->getBasketSizeInPhi();

      if(basketIndex >= halfNumBasketInPhi)             basketIndex -= 2*halfNumBasketInPhi;
      else if(basketIndex <  -1*halfNumBasketInPhi)     basketIndex += 2*halfNumBasketInPhi;

    } else throw(std::runtime_error("\n\nOh No! Calculate_BarrelBasketEnergyFraction called on invalid index.\n\n"));

    indexedBasketEnergy[basketIndex] += (hits->find(*posCurrent))->energy();
  }

  std::vector<double> energyFraction;
  for(std::map<int,double>::iterator posCurrent = indexedBasketEnergy.begin(); posCurrent != indexedBasketEnergy.end(); posCurrent++)
  {
    energyFraction.push_back(indexedBasketEnergy[posCurrent->first]/passedCluster.energy());
  }

  switch(EtaPhi)
  {
    case Eta: energyBasketFractionEta_ = energyFraction; break;
    case Phi: energyBasketFractionPhi_ = energyFraction; break;
  }

}
