#include <cmath>
#include <iostream> // **For testing only remove in final**
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"


//Set Default Values 
bool       ClusterShapeAlgo::param_LogWeighted_(false);
Double32_t ClusterShapeAlgo::param_X0_(0.89);
Double32_t ClusterShapeAlgo::param_T0_(6.2);
Double32_t ClusterShapeAlgo::param_W0_(4.0);

std::map<EBDetId,EcalRecHit> *ClusterShapeAlgo::storedRecHitsMap_ = NULL ;

void ClusterShapeAlgo::Initialize(std::map<std::string,double> providedParameters,
			 std::map<EBDetId,EcalRecHit> *passedRecHitsMap)
{
  std::map<std::string,double>::iterator posCurrent = providedParameters.begin();
 
  if((posCurrent = providedParameters.find("Log_Weighted")) != providedParameters.end())
    param_LogWeighted_ = posCurrent->second;
  if((posCurrent = providedParameters.find("X0")) != providedParameters.end())
     param_X0_ = posCurrent->second;
  if((posCurrent = providedParameters.find("T0")) != providedParameters.end())
     param_T0_ = posCurrent->second;
  if((posCurrent = providedParameters.find("W0")) != providedParameters.end())
     param_W0_ = posCurrent->second;
 
  /*
  if(providedParameters.count("LogWeighted") != 0) 
     param_LogWeighted_ = providedParameters.find("Log_Weighted")->second;
  if(providedParameters.count("X0") != 0) 
     param_X0_ = providedParameters.find("X0")->second;
  if(providedParameters.count("T0") != 0) 
     param_T0_ = providedParameters.find("T0")->second;
  if(providedParameters.count("W0") != 0) 
     param_W0_ = providedParameters.find("W0")->second;
  */

   storedRecHitsMap_ = passedRecHitsMap;
}

reco::ClusterShape ClusterShapeAlgo::Calculate(reco::BasicCluster passedCluster)
{
  if(storedRecHitsMap_ == NULL)
    throw(std::runtime_error("Oh No! ClusterShapeAlgo::Calculate called unitialized."));
   
  ClusterShapeAlgo dataHolder;

  dataHolder.Calculate_TopEnergy(passedCluster);
  dataHolder.Calculate_2ndEnergy(passedCluster);
  dataHolder.Create_Map();
  dataHolder.Calculate_e2x2();
  dataHolder.Calculate_e3x2();
  dataHolder.Calculate_e3x3();
  dataHolder.Calculate_e5x5();
  dataHolder.Calculate_Weights();
  dataHolder.Calculate_eta25phi25();
  dataHolder.Calculate_Covariances();
  dataHolder.Calculate_hadOverEcal();
  
  return reco::ClusterShape( dataHolder.covEtaEta_, 
		  dataHolder.covEtaPhi_,dataHolder.covPhiPhi_, 
                  dataHolder.eMax_, dataHolder.eMaxId_, 
		  dataHolder.e2nd_, dataHolder.e2ndId_,
		  dataHolder.e2x2_, dataHolder.e3x2_, 
		  dataHolder.e3x3_, dataHolder.e5x5_,
                  dataHolder.eta25_,  dataHolder.phi25_,
		  dataHolder.hadOverEcal_ );
}

void ClusterShapeAlgo::Calculate_TopEnergy(reco::BasicCluster passedCluster)
{
  Double32_t eMax=0;
  DetId eMaxId;

  std::vector<DetId> clusterDetIds = passedCluster.getHitsByDetId();
  std::vector<DetId>::iterator posCurrent;

  EcalRecHit testEcalRecHit;

  for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
  {
    testEcalRecHit = ClusterShapeAlgo::storedRecHitsMap_->find(*posCurrent)->second;

    if(testEcalRecHit.energy() > eMax)
    {
      eMax = testEcalRecHit.energy();
      eMaxId = testEcalRecHit.id();
    } 
  }

  eMax_ = eMax;
  eMaxId_ = eMaxId;
}

void ClusterShapeAlgo::Calculate_2ndEnergy(reco::BasicCluster passedCluster)
{
  Double32_t e2nd=0;
  DetId e2ndId;

  std::vector<DetId> clusterDetIds = passedCluster.getHitsByDetId();
  std::vector<DetId>::iterator posCurrent;

  EcalRecHit testEcalRecHit;

  for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
  {
    testEcalRecHit = ClusterShapeAlgo::storedRecHitsMap_->find(*posCurrent)->second;

    if(testEcalRecHit.energy() > e2nd && testEcalRecHit.energy() < eMax_)
    {
      e2nd = testEcalRecHit.energy();
      e2ndId = testEcalRecHit.id();
    } 
  }

  e2nd_ = e2nd;
  e2ndId_ = e2ndId;  
}

void ClusterShapeAlgo::Create_Map()
{
  EcalBarrelHardcodedTopology *barrelTopology = new EcalBarrelHardcodedTopology();
  EcalBarrelNavigator posCurrent(eMaxId_, barrelTopology);

  //Creates Default energyMap_
  for(int i = 0; i <= 4; i++)
    for(int j = 0; j <= 4; j++)
      energyMap_[i][j] = std::make_pair(EBDetId(0), 0);  

  //Fills energyMap_[2][2]
  EcalRecHit tempEcalRecHit = ClusterShapeAlgo::storedRecHitsMap_->find(posCurrent.pos())->second;
  energyMap_[2][2] =  std::make_pair(tempEcalRecHit.id(),tempEcalRecHit.energy());
  
  //Fills energyMap_ along center + 
  for(int direction = 1; direction <= 4; direction++)
  {
    posCurrent.home();

    for(int position = 1; position <=2; position++)
    {
      int x = 0, y=0;

      switch(direction)
      {
        case 1: posCurrent.north(); x=0; y=-position; break;
	case 2: posCurrent.east();  x=position; y=0;  break;
	case 3: posCurrent.south(); x=0; y=position;  break;
        case 4: posCurrent.west();  x=-position; y=0; break;  
      }

      if(posCurrent.pos() == EBDetId(0)) break;

      tempEcalRecHit = ClusterShapeAlgo::storedRecHitsMap_->find(posCurrent.pos())->second;
      energyMap_[2+x][2+y] =  std::make_pair(tempEcalRecHit.id(),tempEcalRecHit.energy());
    }
  }

  //CREATES A TEST MAP -- FUNCTION NOT IMPLEMENTED 

  for(int i = 0; i <= 4; i++)
    for(int j = 0; j <= 4; j++)
      energyMap_[i][j] = std::make_pair(EBDetId(0), i+j);  
  
}

void ClusterShapeAlgo::Calculate_e2x2()
{
  Double32_t e2x2Max = 0;
  Double32_t e2x2Test = 0;

  e2x2Test  = energyMap_[1][1].second;
  e2x2Test += energyMap_[1][2].second;
  e2x2Test += energyMap_[2][1].second;
  e2x2Test += energyMap_[2][2].second;

  e2x2Max = (e2x2Test > e2x2Max) ? e2x2Test : e2x2Max;

  e2x2Test  = energyMap_[2][1].second;
  e2x2Test += energyMap_[3][1].second;
  e2x2Test += energyMap_[2][2].second;
  e2x2Test += energyMap_[3][2].second;

  e2x2Max = (e2x2Test > e2x2Max) ? e2x2Test : e2x2Max;

  e2x2Test  = energyMap_[1][2].second;
  e2x2Test += energyMap_[2][2].second;
  e2x2Test += energyMap_[1][3].second;
  e2x2Test += energyMap_[2][3].second;

  e2x2Max = (e2x2Test > e2x2Max) ? e2x2Test : e2x2Max;

  e2x2Test  = energyMap_[2][2].second;
  e2x2Test += energyMap_[3][2].second;
  e2x2Test += energyMap_[2][3].second;
  e2x2Test += energyMap_[3][3].second;

  e2x2Max = (e2x2Test > e2x2Max) ? e2x2Test : e2x2Max;

  e2x2_ = e2x2Max;

}

void ClusterShapeAlgo::Calculate_e3x2(){} 

void ClusterShapeAlgo::Calculate_e3x3()
{
  Double32_t e3x3=0; 

  for(int i = 1; i <= 3; i++)
    for(int j = 1; j <= 3; j++)
      e3x3 += energyMap_[i][j].second;

  e3x3_ = e3x3;

}

void ClusterShapeAlgo::Calculate_e5x5()
{
  Double32_t e5x5=0; 

  for(int i = 0; i <= 4; i++)
    for(int j = 0; j <= 4; j++)
      e5x5 += energyMap_[i][j].second;

  e5x5_ = e5x5;

}

void ClusterShapeAlgo::Calculate_Weights()
{
  //NOT CHECKED FOR CORRECTNESS AT THIS POINT


  weightsTotal_ = 0;
  
  if(param_LogWeighted_)
  {
    for(int i = 0; i <= 4; i++)
      for(int j = 0; j <= 4; j++)
	{
	  weightsMap_[i][j] = (energyMap_[i][j].second == 0) 
	    ? 0 : (param_W0_ + log(energyMap_[i][j].second/e5x5_) <= 0) 
	    ? 0 : param_W0_ + log(energyMap_[i][j].second/e5x5_);
	  weightsTotal_ += weightsMap_[i][j];
	} 
  }
  else
  {
    weightsTotal_ = 1;

    for(int i = 0; i <= 4; i++)
      for(int j = 0; j <= 4; j++)
	weightsMap_[i][j] = energyMap_[i][j].second/e5x5_;
  }

}

void ClusterShapeAlgo::Calculate_eta25phi25(){}

void ClusterShapeAlgo::Calculate_Covariances(){}

void ClusterShapeAlgo::Calculate_hadOverEcal(){}
