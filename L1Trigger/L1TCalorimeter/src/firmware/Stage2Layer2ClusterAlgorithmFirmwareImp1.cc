///
/// \class l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2ClusterAlgorithmFirmware.h"
//#include "DataFormats/Math/interface/LorentzVector.h "

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

/*****************************************************************/
l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::Stage2Layer2ClusterAlgorithmFirmwareImp1(CaloParamsHelper* params, ClusterInput clusterInput) :
  clusterInput_(clusterInput),
  seedThreshold_(1),
  clusterThreshold_(1),
  hcalThreshold_(1),
  params_(params)
/*****************************************************************/
{

}


/*****************************************************************/
l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::~Stage2Layer2ClusterAlgorithmFirmwareImp1()
/*****************************************************************/
{
}


/*****************************************************************/
void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower>& towers, std::vector<l1t::CaloCluster>& clusters)
/*****************************************************************/
{
  if (clusterInput_==E)
  {
    seedThreshold_    = floor(params_->egSeedThreshold()/params_->towerLsbE());
    clusterThreshold_ = floor(params_->egNeighbourThreshold()/params_->towerLsbE());
  }
  else if (clusterInput_==EH)
  {
    seedThreshold_    = floor(params_->egSeedThreshold()/params_->towerLsbSum());
    clusterThreshold_ = floor(params_->egNeighbourThreshold()/params_->towerLsbSum());
  }
  if (clusterInput_==H)
  {
    seedThreshold_    = floor(params_->egSeedThreshold()/params_->towerLsbH());
    clusterThreshold_ = floor(params_->egNeighbourThreshold()/params_->towerLsbH());
  }

  hcalThreshold_ = floor(params_->egHcalThreshold()/params_->towerLsbH());

  clustering(towers, clusters);
  filtering (towers, clusters);
  //No sharing in new implementation
  //sharing   (towers, clusters);
  refining  (towers, clusters);
}

/*****************************************************************/
void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::clustering(const std::vector<l1t::CaloTower>& towers, std::vector<l1t::CaloCluster>& clusters)
/*****************************************************************/
{
  // navigator
  l1t::CaloStage2Nav caloNav;

  // Build clusters passing seed threshold
  for(const auto& tower : towers)
  {
    int iEta = tower.hwEta();
    int iPhi = tower.hwPhi();
    int hwEt = 0;
    if(clusterInput_==E)       hwEt = tower.hwEtEm();
    else if(clusterInput_==EH) hwEt = tower.hwPt();
    else if(clusterInput_==H)  hwEt = tower.hwEtHad();
    int hwEtEm  = tower.hwEtEm();
    int hwEtHad = tower.hwEtHad();
    // Check if the seed tower pass the seed threshold
    if(hwEt>=seedThreshold_)
    {
      math::XYZTLorentzVector emptyP4;
      clusters.push_back( l1t::CaloCluster(emptyP4, hwEt, iEta, iPhi) );
      l1t::CaloCluster& cluster = clusters.back();
      cluster.setHwPtEm(hwEtEm);
      cluster.setHwPtHad(hwEtHad);
      cluster.setHwSeedPt(hwEt);
      
      bool hOverE = idHoverE(tower);
      cluster.setHOverE(hOverE);
      // FG of the cluster is FG of the seed
      bool fg = (tower.hwQual() & (0x1<<3));
      cluster.setFgECAL((int)fg);
    }
  }


  // check if neighbour towers are below clustering threshold
  for(auto& cluster : clusters)
  {
    if( cluster.isValid() )
    {
      // look at the energies in neighbour towers
      int iEta   = cluster.hwEta();
      int iPhi   = cluster.hwPhi();
      int iEtaP  = caloNav.offsetIEta(iEta,  1);
      int iEtaM  = caloNav.offsetIEta(iEta, -1);
      int iPhiP  = caloNav.offsetIPhi(iPhi,  1);
      int iPhiP2 = caloNav.offsetIPhi(iPhi,  2);
      int iPhiM  = caloNav.offsetIPhi(iPhi, -1);
      int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
      const l1t::CaloTower& towerNW = l1t::CaloTools::getTower(towers, iEtaM, iPhiM);
      const l1t::CaloTower& towerN  = l1t::CaloTools::getTower(towers, iEta , iPhiM);
      const l1t::CaloTower& towerNE = l1t::CaloTools::getTower(towers, iEtaP, iPhiM);
      const l1t::CaloTower& towerE  = l1t::CaloTools::getTower(towers, iEtaP, iPhi );
      const l1t::CaloTower& towerSE = l1t::CaloTools::getTower(towers, iEtaP, iPhiP);
      const l1t::CaloTower& towerS  = l1t::CaloTools::getTower(towers, iEta , iPhiP);
      const l1t::CaloTower& towerSW = l1t::CaloTools::getTower(towers, iEtaM, iPhiP);
      const l1t::CaloTower& towerW  = l1t::CaloTools::getTower(towers, iEtaM, iPhi );
      const l1t::CaloTower& towerNN = l1t::CaloTools::getTower(towers, iEta , iPhiM2);
      const l1t::CaloTower& towerSS = l1t::CaloTools::getTower(towers, iEta , iPhiP2);
      int towerEtNW = 0;
      int towerEtN  = 0;
      int towerEtNE = 0;
      int towerEtE  = 0;
      int towerEtSE = 0;
      int towerEtS  = 0;
      int towerEtSW = 0;
      int towerEtW  = 0;
      int towerEtNN = 0;
      int towerEtSS = 0;
      if(clusterInput_==E)
      {
        towerEtNW = towerNW.hwEtEm();
        towerEtN  = towerN .hwEtEm();
        towerEtNE = towerNE.hwEtEm();
        towerEtE  = towerE .hwEtEm();
        towerEtSE = towerSE.hwEtEm();
        towerEtS  = towerS .hwEtEm();
        towerEtSW = towerSW.hwEtEm();
        towerEtW  = towerW .hwEtEm();
        towerEtNN = towerNN.hwEtEm();
        towerEtSS = towerSS.hwEtEm();
      }
      else if(clusterInput_==EH)
      {
	towerEtNW = towerNW.hwPt();
        towerEtN  = towerN .hwPt();
        towerEtNE = towerNE.hwPt();
        towerEtE  = towerE .hwPt();
        towerEtSE = towerSE.hwPt();
        towerEtS  = towerS .hwPt();
        towerEtSW = towerSW.hwPt();
        towerEtW  = towerW .hwPt();
        towerEtNN = towerNN.hwPt();
        towerEtSS = towerSS.hwPt();
      }
      else if(clusterInput_==H)
      {
        towerEtNW = towerNW.hwEtHad();
        towerEtN  = towerN .hwEtHad();
        towerEtNE = towerNE.hwEtHad();
        towerEtE  = towerE .hwEtHad();
        towerEtSE = towerSE.hwEtHad();
        towerEtS  = towerS .hwEtHad();
        towerEtSW = towerSW.hwEtHad();
        towerEtW  = towerW .hwEtHad();
        towerEtNN = towerNN.hwEtHad();
        towerEtSS = towerSS.hwEtHad();
      }

      // check which towers can be clustered to the seed
      if(towerEtNW < clusterThreshold_) cluster.setClusterFlag(CaloCluster::INCLUDE_NW, false);
      if(towerEtN  < clusterThreshold_)
      {
        cluster.setClusterFlag(CaloCluster::INCLUDE_N , false);
        cluster.setClusterFlag(CaloCluster::INCLUDE_NN, false);
      }
      if(towerEtNE < clusterThreshold_) cluster.setClusterFlag(CaloCluster::INCLUDE_NE, false);
      if(towerEtE  < clusterThreshold_) cluster.setClusterFlag(CaloCluster::INCLUDE_E , false);
      if(towerEtSE < clusterThreshold_) cluster.setClusterFlag(CaloCluster::INCLUDE_SE, false);
      if(towerEtS  < clusterThreshold_)
      {
        cluster.setClusterFlag(CaloCluster::INCLUDE_S , false);
        cluster.setClusterFlag(CaloCluster::INCLUDE_SS, false);
      }
      if(towerEtSW < clusterThreshold_) cluster.setClusterFlag(CaloCluster::INCLUDE_SW, false);
      if(towerEtW  < clusterThreshold_) cluster.setClusterFlag(CaloCluster::INCLUDE_W , false);
      if(towerEtNN < clusterThreshold_) cluster.setClusterFlag(CaloCluster::INCLUDE_NN, false);
      if(towerEtSS < clusterThreshold_) cluster.setClusterFlag(CaloCluster::INCLUDE_SS, false);

    }
  }

}



/*****************************************************************/
void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::filtering(const std::vector<l1t::CaloTower>& towers, std::vector<l1t::CaloCluster>& clusters)
/*****************************************************************/
{


  // adapted from jet overlap filtering

  int mask[9][3] = {
    { 2,2,2 },
    { 2,2,2 },
    { 2,2,2 },
    { 1,2,2 },
    { 1,0,2 },
    { 1,1,2 },
    { 1,1,1 },
    { 1,1,1 },
    { 1,1,1 },
  };

  // navigator
  l1t::CaloStage2Nav caloNav;

  // Filter: keep only local maxima in a 9x3 region
  // If two neighbor seeds have the same energy, favor the most central one
  for(auto& cluster : clusters)
  {
    // retrieve neighbour cluster candidates. At this stage they only contain the seed tower.
    int iEta   = cluster.hwEta();
    int iPhi   = cluster.hwPhi();
    bool filter = false;
    for( int deta = -1; deta < 2; ++deta ) 
    {
	  for( int dphi = -4; dphi < 5; ++dphi )
      {
          int iEtaNeigh = caloNav.offsetIEta(iEta,  deta);
          int iPhiNeigh = caloNav.offsetIPhi(iPhi,  dphi);
          const l1t::CaloCluster& clusterNeigh = l1t::CaloTools::getCluster(clusters, iEtaNeigh, iPhiNeigh);
	  	    

          if      (mask[8-(dphi+4)][deta+1] == 0) continue;
          else if (mask[8-(dphi+4)][deta+1] == 1) filter = (clusterNeigh.hwPt() >   cluster.hwPt());
          else if (mask[8-(dphi+4)][deta+1] == 2) filter = (clusterNeigh.hwPt() >=  cluster.hwPt());
          if(filter) 
          {
              cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
              break;
          }
      }
      if(filter) break;
    }
  }

}



/*****************************************************************/
void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::refining(const std::vector<l1t::CaloTower>& towers, std::vector<l1t::CaloCluster>& clusters)
/*****************************************************************/
{
  // navigator
  l1t::CaloStage2Nav caloNav;

  // trim cluster
  for(auto& cluster : clusters)
  {
    if( cluster.isValid() )
    {
      int iEta = cluster.hwEta();
      int iPhi = cluster.hwPhi();
      int iEtaP  = caloNav.offsetIEta(iEta, 1);
      int iEtaM  = caloNav.offsetIEta(iEta, -1);
      int iPhiP  = caloNav.offsetIPhi(iPhi, 1);
      int iPhiP2 = caloNav.offsetIPhi(iPhi, 2);
      int iPhiM  = caloNav.offsetIPhi(iPhi, -1);
      int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
      const l1t::CaloTower& towerNW = l1t::CaloTools::getTower(towers, iEtaM, iPhiM);
      const l1t::CaloTower& towerN  = l1t::CaloTools::getTower(towers, iEta , iPhiM);
      const l1t::CaloTower& towerNE = l1t::CaloTools::getTower(towers, iEtaP, iPhiM);
      const l1t::CaloTower& towerE  = l1t::CaloTools::getTower(towers, iEtaP, iPhi );
      const l1t::CaloTower& towerSE = l1t::CaloTools::getTower(towers, iEtaP, iPhiP);
      const l1t::CaloTower& towerS  = l1t::CaloTools::getTower(towers, iEta , iPhiP);
      const l1t::CaloTower& towerSW = l1t::CaloTools::getTower(towers, iEtaM, iPhiP);
      const l1t::CaloTower& towerW  = l1t::CaloTools::getTower(towers, iEtaM, iPhi );
      const l1t::CaloTower& towerNN = l1t::CaloTools::getTower(towers, iEta , iPhiM2);
      const l1t::CaloTower& towerSS = l1t::CaloTools::getTower(towers, iEta , iPhiP2);

      int towerEtNW = 0;
      int towerEtN  = 0;
      int towerEtNE = 0;
      int towerEtE  = 0;
      int towerEtSE = 0;
      int towerEtS  = 0;
      int towerEtSW = 0;
      int towerEtW  = 0;
      int towerEtNN = 0;
      int towerEtSS = 0;
      if(clusterInput_==E)
      {
        towerEtNW = towerNW.hwEtEm();
        towerEtN  = towerN .hwEtEm();
        towerEtNE = towerNE.hwEtEm();
        towerEtE  = towerE .hwEtEm();
        towerEtSE = towerSE.hwEtEm();
        towerEtS  = towerS .hwEtEm();
        towerEtSW = towerSW.hwEtEm();
        towerEtW  = towerW .hwEtEm();
        towerEtNN = towerNN.hwEtEm();
        towerEtSS = towerSS.hwEtEm();
      }
      else if(clusterInput_==EH)
      {
	towerEtNW = towerNW.hwPt();
        towerEtN  = towerN .hwPt();
        towerEtNE = towerNE.hwPt();
        towerEtE  = towerE .hwPt();
        towerEtSE = towerSE.hwPt();
        towerEtS  = towerS .hwPt();
        towerEtSW = towerSW.hwPt();
        towerEtW  = towerW .hwPt();
        towerEtNN = towerNN.hwPt();
        towerEtSS = towerSS.hwPt();

      }
      else if(clusterInput_==H)
      {
        towerEtNW = towerNW.hwEtHad();
        towerEtN  = towerN .hwEtHad();
        towerEtNE = towerNE.hwEtHad();
        towerEtE  = towerE .hwEtHad();
        towerEtSE = towerSE.hwEtHad();
        towerEtS  = towerS .hwEtHad();
        towerEtSW = towerSW.hwEtHad();
        towerEtW  = towerW .hwEtHad();
        towerEtNN = towerNN.hwEtHad();
        towerEtSS = towerSS.hwEtHad();
      }

      int towerEtEmNW = towerNW.hwEtEm();
      int towerEtEmN  = towerN .hwEtEm();
      int towerEtEmNE = towerNE.hwEtEm();
      int towerEtEmE  = towerE .hwEtEm();
      int towerEtEmSE = towerSE.hwEtEm();
      int towerEtEmS  = towerS .hwEtEm();
      int towerEtEmSW = towerSW.hwEtEm();
      int towerEtEmW  = towerW .hwEtEm();
      int towerEtEmNN = towerNN.hwEtEm();
      int towerEtEmSS = towerSS.hwEtEm();
      //
      int towerEtHadNW = towerNW.hwEtHad();
      int towerEtHadN  = towerN .hwEtHad();
      int towerEtHadNE = towerNE.hwEtHad();
      int towerEtHadE  = towerE .hwEtHad();
      int towerEtHadSE = towerSE.hwEtHad();
      int towerEtHadS  = towerS .hwEtHad();
      int towerEtHadSW = towerSW.hwEtHad();
      int towerEtHadW  = towerW .hwEtHad();
      int towerEtHadNN = towerNN.hwEtHad();
      int towerEtHadSS = towerSS.hwEtHad();

      // trim one eta-side
      // The side with largest energy will be kept
      int EtEtaRight = 0;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NE)) EtEtaRight += towerEtNE;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_E))  EtEtaRight += towerEtE;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SE)) EtEtaRight += towerEtSE;
      int EtEtaLeft  = 0;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NW)) EtEtaLeft += towerEtNW;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_W))  EtEtaLeft += towerEtW;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SW)) EtEtaLeft += towerEtSW;

      cluster.setClusterFlag(CaloCluster::TRIM_LEFT, (EtEtaRight>= EtEtaLeft) );


      if(cluster.checkClusterFlag(CaloCluster::TRIM_LEFT))
      {
        cluster.setClusterFlag(CaloCluster::INCLUDE_NW, false);
        cluster.setClusterFlag(CaloCluster::INCLUDE_W , false);
        cluster.setClusterFlag(CaloCluster::INCLUDE_SW, false);
      }
      else
      {
        cluster.setClusterFlag(CaloCluster::INCLUDE_NE, false);
        cluster.setClusterFlag(CaloCluster::INCLUDE_E , false);
        cluster.setClusterFlag(CaloCluster::INCLUDE_SE, false);
      }


      // compute cluster energy according to cluster flags
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NW)) cluster.setHwPt(cluster.hwPt() + towerEtNW);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_N))  cluster.setHwPt(cluster.hwPt() + towerEtN);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NE)) cluster.setHwPt(cluster.hwPt() + towerEtNE);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_E))  cluster.setHwPt(cluster.hwPt() + towerEtE);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SE)) cluster.setHwPt(cluster.hwPt() + towerEtSE);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_S))  cluster.setHwPt(cluster.hwPt() + towerEtS);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SW)) cluster.setHwPt(cluster.hwPt() + towerEtSW);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_W))  cluster.setHwPt(cluster.hwPt() + towerEtW);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NN)) cluster.setHwPt(cluster.hwPt() + towerEtNN);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SS)) cluster.setHwPt(cluster.hwPt() + towerEtSS);
      //
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NW)) cluster.setHwPtEm(cluster.hwPtEm() + towerEtEmNW);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_N))  cluster.setHwPtEm(cluster.hwPtEm() + towerEtEmN);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NE)) cluster.setHwPtEm(cluster.hwPtEm() + towerEtEmNE);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_E))  cluster.setHwPtEm(cluster.hwPtEm() + towerEtEmE);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SE)) cluster.setHwPtEm(cluster.hwPtEm() + towerEtEmSE);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_S))  cluster.setHwPtEm(cluster.hwPtEm() + towerEtEmS);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SW)) cluster.setHwPtEm(cluster.hwPtEm() + towerEtEmSW);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_W))  cluster.setHwPtEm(cluster.hwPtEm() + towerEtEmW);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NN)) cluster.setHwPtEm(cluster.hwPtEm() + towerEtEmNN);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SS)) cluster.setHwPtEm(cluster.hwPtEm() + towerEtEmSS);
      //
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NW)) cluster.setHwPtHad(cluster.hwPtHad() + towerEtHadNW);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_N))  cluster.setHwPtHad(cluster.hwPtHad() + towerEtHadN);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NE)) cluster.setHwPtHad(cluster.hwPtHad() + towerEtHadNE);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_E))  cluster.setHwPtHad(cluster.hwPtHad() + towerEtHadE);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SE)) cluster.setHwPtHad(cluster.hwPtHad() + towerEtHadSE);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_S))  cluster.setHwPtHad(cluster.hwPtHad() + towerEtHadS);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SW)) cluster.setHwPtHad(cluster.hwPtHad() + towerEtHadSW);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_W))  cluster.setHwPtHad(cluster.hwPtHad() + towerEtHadW);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NN)) cluster.setHwPtHad(cluster.hwPtHad() + towerEtHadNN);
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SS)) cluster.setHwPtHad(cluster.hwPtHad() + towerEtHadSS);


      // Compute fine-grain position within the seed tower,
      // according to the distribution of energy in the cluster
      int fgEta = 0;
      int fgPhi = 0;
      
      if(EtEtaRight>EtEtaLeft) fgEta = 2;
      else if(EtEtaLeft>EtEtaRight) fgEta = 1;

      int EtUp   = 0;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NE)) EtUp += towerEtNE;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_N))  EtUp += towerEtN;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NW)) EtUp += towerEtNW;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NN)) EtUp += towerEtNN;
      int EtDown = 0;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SE)) EtDown += towerEtSE;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_S))  EtDown += towerEtS;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SW)) EtDown += towerEtSW;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SS)) EtDown += towerEtSS;
      //
      if(EtDown>EtUp) fgPhi = 2;
      else if(EtUp>EtDown) fgPhi = 1;
      //
      cluster.setFgEta(fgEta);
      cluster.setFgPhi(fgPhi);
    }
  }
}







bool l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::idHoverE(const l1t::CaloTower tow){

  bool hOverEBit = true;

  int ratio =  tow.hwEtRatio();
  int qual  = tow.hwQual();
  bool denomZeroFlag = ((qual&0x1) > 0);
  bool eOverHFlag    = ((qual&0x2) > 0);

  if (denomZeroFlag && !eOverHFlag)
    hOverEBit = false;
  if (denomZeroFlag && eOverHFlag)
    hOverEBit = true;
  if (!denomZeroFlag && !eOverHFlag) // H > E, ratio = log(H/E)
    hOverEBit = false;
  if (!denomZeroFlag && eOverHFlag){ // E >= H , so ratio==log(E/H)
    if(abs(tow.hwEta())< 16 )
      hOverEBit = ratio >= 5;
    else
    hOverEBit = ratio >= 4;
  }
  
  return hOverEBit;

}
