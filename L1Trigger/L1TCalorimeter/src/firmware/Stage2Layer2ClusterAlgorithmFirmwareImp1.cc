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

#include "CondFormats/L1TObjects/interface/CaloParams.h"

l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::Stage2Layer2ClusterAlgorithmFirmwareImp1(CaloParams* params, ClusterInput clusterInput) :
  m_clusterInput(clusterInput),
  m_trimCorners(true),
  params_(params)
{

  if (m_clusterInput==E) {
    m_seedThreshold    = floor(params_->egSeedThreshold()/params_->towerLsbE()); 
    m_clusterThreshold = floor(params_->egNeighbourThreshold()/params_->towerLsbE());
  }
  if (m_clusterInput==H) {
    m_seedThreshold    = floor(params_->egSeedThreshold()/params_->towerLsbH()); 
    m_clusterThreshold = floor(params_->egNeighbourThreshold()/params_->towerLsbH());
  }
  else {
    m_seedThreshold    = floor(params_->egSeedThreshold()/params_->towerLsbSum()); 
    m_clusterThreshold = floor(params_->egNeighbourThreshold()/params_->towerLsbSum());
  }
  m_hcalThreshold = floor(params_->egHcalThreshold()/params_->towerLsbH());
}


l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::~Stage2Layer2ClusterAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
							      std::vector<l1t::CaloCluster> & clusters) {

  clustering(towers, clusters);
  filtering(towers, clusters);
  sharing(towers, clusters);
  refining(towers, clusters);
}

void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::clustering(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters){
  // navigator
  l1t::CaloStage2Nav caloNav;

  // Build clusters passing seed threshold
  for(size_t towerNr=0;towerNr<towers.size();towerNr++){
    int iEta = towers[towerNr].hwEta();
    int iPhi = towers[towerNr].hwPhi();
    int hwEt = 0;
    if(m_clusterInput==E)       hwEt = towers[towerNr].hwEtEm();
    else if(m_clusterInput==H)  hwEt = towers[towerNr].hwEtHad();
    else if(m_clusterInput==EH) hwEt = towers[towerNr].hwEtEm() + towers[towerNr].hwEtHad();
    int hwEtEm  = towers[towerNr].hwEtEm();
    int hwEtHad = towers[towerNr].hwEtHad();
    if(hwEt>=m_seedThreshold)
    {
      math::XYZTLorentzVector emptyP4;
      clusters.push_back( CaloCluster(emptyP4, hwEt, iEta, iPhi) );
      clusters.back().setHwPtEm(hwEtEm);
      clusters.back().setHwPtHad(hwEtHad);
      clusters.back().setHwSeedPt(hwEt);
      // H/E of the cluster is H/E of the seed
      int hwEtHadTh = (towers[towerNr].hwEtHad()>=m_hcalThreshold ? towers[towerNr].hwEtHad() : 0);
      int hOverE = (towers[towerNr].hwEtEm()>0 ? (hwEtHadTh<<8)/towers[towerNr].hwEtEm() : 255);
      if(hOverE>255) hOverE = 255; // bound H/E at 1-? In the future it will be useful to replace with H/(E+H) (or add an other variable), for taus.
      clusters.back().setHOverE(hOverE);
      // FG of the cluster is FG of the seed
      bool fg = (towers[towerNr].hwQual() & (0x1<<2));
      clusters.back().setFgECAL((int)fg);
    }
  }


  // check if neighbour towers are below clustering threshold
  for(size_t clusterNr=0;clusterNr<clusters.size();clusterNr++){
    l1t::CaloCluster& cluster = clusters[clusterNr];
    if( cluster.isValid() ){
      int iEta = cluster.hwEta();
      int iPhi = cluster.hwPhi();
      int iEtaP = caloNav.offsetIEta(iEta, 1);
      int iEtaM = caloNav.offsetIEta(iEta, -1);
      int iPhiP = caloNav.offsetIPhi(iPhi, 1);
      int iPhiP2 = caloNav.offsetIPhi(iPhi, 2);
      int iPhiM = caloNav.offsetIPhi(iPhi, -1);
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
      if(m_clusterInput==E){
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
      else if(m_clusterInput==H){
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
      else if(m_clusterInput==EH){
        towerEtNW = towerNW.hwEtEm() + towerNW.hwEtHad();
        towerEtN  = towerN .hwEtEm() + towerN .hwEtHad();
        towerEtNE = towerNE.hwEtEm() + towerNE.hwEtHad();
        towerEtE  = towerE .hwEtEm() + towerE .hwEtHad();
        towerEtSE = towerSE.hwEtEm() + towerSE.hwEtHad();
        towerEtS  = towerS .hwEtEm() + towerS .hwEtHad();
        towerEtSW = towerSW.hwEtEm() + towerSW.hwEtHad();
        towerEtW  = towerW .hwEtEm() + towerW .hwEtHad();
        towerEtNN = towerNN.hwEtEm() + towerNN.hwEtHad();
        towerEtSS = towerSS.hwEtEm() + towerSS.hwEtHad();
      }

      if(towerEtNW < m_clusterThreshold) cluster.setClusterFlag(CaloCluster::INCLUDE_NW, false);
      if(towerEtN  < m_clusterThreshold){
        cluster.setClusterFlag(CaloCluster::INCLUDE_N , false);
        cluster.setClusterFlag(CaloCluster::INCLUDE_NN, false);
      }
      if(towerEtNE < m_clusterThreshold) cluster.setClusterFlag(CaloCluster::INCLUDE_NE, false);
      if(towerEtE  < m_clusterThreshold) cluster.setClusterFlag(CaloCluster::INCLUDE_E , false);
      if(towerEtSE < m_clusterThreshold) cluster.setClusterFlag(CaloCluster::INCLUDE_SE, false);
      if(towerEtS  < m_clusterThreshold){
        cluster.setClusterFlag(CaloCluster::INCLUDE_S , false);
        cluster.setClusterFlag(CaloCluster::INCLUDE_SS, false);
      }
      if(towerEtSW < m_clusterThreshold) cluster.setClusterFlag(CaloCluster::INCLUDE_SW, false);
      if(towerEtW  < m_clusterThreshold) cluster.setClusterFlag(CaloCluster::INCLUDE_W , false);
      if(towerEtNN < m_clusterThreshold) cluster.setClusterFlag(CaloCluster::INCLUDE_NN, false);
      if(towerEtSS < m_clusterThreshold) cluster.setClusterFlag(CaloCluster::INCLUDE_SS, false);

    }
  }

}


void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::filtering(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters){
  // navigator
  l1t::CaloStage2Nav caloNav;

  // Filter: keep only local maxima
  // If two neighbor seeds have the same energy, favor the most central one
  for(size_t clusterNr=0;clusterNr<clusters.size();clusterNr++){
    l1t::CaloCluster& cluster = clusters[clusterNr];
    int iEta = cluster.hwEta();
    int iPhi = cluster.hwPhi();
    int iEtaP = caloNav.offsetIEta(iEta, 1);
    int iEtaM = caloNav.offsetIEta(iEta, -1);
    int iPhiP = caloNav.offsetIPhi(iPhi, 1);
    int iPhiP2 = caloNav.offsetIPhi(iPhi, 2);
    int iPhiM  = caloNav.offsetIPhi(iPhi, -1);
    int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
    const l1t::CaloCluster& clusterNW = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiM);
    const l1t::CaloCluster& clusterN  = l1t::CaloTools::getCluster(clusters, iEta , iPhiM);
    const l1t::CaloCluster& clusterNE = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiM);
    const l1t::CaloCluster& clusterE  = l1t::CaloTools::getCluster(clusters, iEtaP, iPhi );
    const l1t::CaloCluster& clusterSE = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiP);
    const l1t::CaloCluster& clusterS  = l1t::CaloTools::getCluster(clusters, iEta , iPhiP);
    const l1t::CaloCluster& clusterSW = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiP);
    const l1t::CaloCluster& clusterW  = l1t::CaloTools::getCluster(clusters, iEtaM, iPhi );
    const l1t::CaloCluster& clusterNN = l1t::CaloTools::getCluster(clusters, iEta , iPhiM2);
    const l1t::CaloCluster& clusterSS = l1t::CaloTools::getCluster(clusters, iEta , iPhiP2);

    const l1t::CaloTower& towerN  = l1t::CaloTools::getTower(towers, iEta , iPhiM);
    const l1t::CaloTower& towerS  = l1t::CaloTools::getTower(towers, iEta , iPhiP);
    int towerEtN  = 0;
    int towerEtS  = 0;
    if(m_clusterInput==E){
      towerEtN  = towerN .hwEtEm();
      towerEtS  = towerS .hwEtEm();
    }
    else if(m_clusterInput==H){
      towerEtN  = towerN .hwEtHad();
      towerEtS  = towerS .hwEtHad();
    }
    else if(m_clusterInput==EH){
      towerEtN  = towerN .hwEtEm() + towerN .hwEtHad();
      towerEtS  = towerS .hwEtEm() + towerS .hwEtHad();
    }

    if(iEta>1){
      if(clusterNW.hwPt() >= cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterN .hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterNE.hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterE .hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterSE.hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterS .hwPt() >= cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterSW.hwPt() >= cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterW .hwPt() >= cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      // NOT_IN_FIRMWARE
      if(towerEtN >= m_clusterThreshold && clusterNN.hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(towerEtS >= m_clusterThreshold && clusterSS.hwPt() >= cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      // END NOT_IN_FIRMWARE
    }
    else if(iEta<0){
      if(clusterNW.hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterN .hwPt() >= cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterNE.hwPt() >= cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterE .hwPt() >= cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterSE.hwPt() >= cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterS .hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterSW.hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterW .hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      // NOT_IN_FIRMWARE
      if(towerEtN >= m_clusterThreshold && clusterNN.hwPt() >= cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(towerEtS >= m_clusterThreshold && clusterSS.hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      // END NOT_IN_FIRMWARE
    }
    else{ // iEta==1
      if(clusterNW.hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterN .hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterNE.hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterE .hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterSE.hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterS .hwPt() >= cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterSW.hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(clusterW .hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      // NOT_IN_FIRMWARE
      if(towerEtN >= m_clusterThreshold && clusterNN.hwPt() >  cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      if(towerEtS >= m_clusterThreshold && clusterSS.hwPt() >= cluster.hwPt()) cluster.setClusterFlag(CaloCluster::INCLUDE_SEED, false);
      // END NOT_IN_FIRMWARE
    }
  }

}


void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::sharing(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters){
  // navigator
  l1t::CaloStage2Nav caloNav;

  // Share tower energies between clusters
  for(size_t clusterNr=0;clusterNr<clusters.size();clusterNr++){
    l1t::CaloCluster& cluster = clusters[clusterNr];
    if( cluster.isValid() ){
      int iEta = cluster.hwEta();
      int iPhi = cluster.hwPhi();
      int iEtaP  = caloNav.offsetIEta(iEta, 1);
      int iEtaP2 = caloNav.offsetIEta(iEta, 2);
      int iEtaM  = caloNav.offsetIEta(iEta, -1);
      int iEtaM2 = caloNav.offsetIEta(iEta, -2);
      int iPhiP  = caloNav.offsetIPhi(iPhi, 1);
      int iPhiP2 = caloNav.offsetIPhi(iPhi, 2);
      int iPhiP3 = caloNav.offsetIPhi(iPhi, 3);
      int iPhiP4 = caloNav.offsetIPhi(iPhi, 4);
      int iPhiM  = caloNav.offsetIPhi(iPhi, -1);
      int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
      int iPhiM3 = caloNav.offsetIPhi(iPhi, -3);
      int iPhiM4 = caloNav.offsetIPhi(iPhi, -4);
      const l1t::CaloCluster& clusterNNWW = l1t::CaloTools::getCluster(clusters, iEtaM2, iPhiM2);
      const l1t::CaloCluster& clusterNNW  = l1t::CaloTools::getCluster(clusters, iEtaM , iPhiM2);
      const l1t::CaloCluster& clusterNN   = l1t::CaloTools::getCluster(clusters, iEta  , iPhiM2);
      const l1t::CaloCluster& clusterNNE  = l1t::CaloTools::getCluster(clusters, iEtaP , iPhiM2);
      const l1t::CaloCluster& clusterNNEE = l1t::CaloTools::getCluster(clusters, iEtaP2, iPhiM2);
      const l1t::CaloCluster& clusterNEE  = l1t::CaloTools::getCluster(clusters, iEtaP2, iPhiM);
      const l1t::CaloCluster& clusterEE   = l1t::CaloTools::getCluster(clusters, iEtaP2, iPhi);
      const l1t::CaloCluster& clusterSEE  = l1t::CaloTools::getCluster(clusters, iEtaP2, iPhiP);
      const l1t::CaloCluster& clusterSSEE = l1t::CaloTools::getCluster(clusters, iEtaP2, iPhiP2);
      const l1t::CaloCluster& clusterSSE  = l1t::CaloTools::getCluster(clusters, iEtaP , iPhiP2);
      const l1t::CaloCluster& clusterSS   = l1t::CaloTools::getCluster(clusters, iEta  , iPhiP2);
      const l1t::CaloCluster& clusterSSW  = l1t::CaloTools::getCluster(clusters, iEtaM , iPhiP2);
      const l1t::CaloCluster& clusterSSWW = l1t::CaloTools::getCluster(clusters, iEtaM2, iPhiP2);
      const l1t::CaloCluster& clusterSWW  = l1t::CaloTools::getCluster(clusters, iEtaM2, iPhiP);
      const l1t::CaloCluster& clusterWW   = l1t::CaloTools::getCluster(clusters, iEtaM2, iPhi);
      const l1t::CaloCluster& clusterNWW  = l1t::CaloTools::getCluster(clusters, iEtaM2, iPhiM);
      const l1t::CaloCluster& clusterNNNW = l1t::CaloTools::getCluster(clusters, iEtaM , iPhiM3);
      const l1t::CaloCluster& clusterNNN  = l1t::CaloTools::getCluster(clusters, iEta  , iPhiM3);
      const l1t::CaloCluster& clusterNNNE = l1t::CaloTools::getCluster(clusters, iEtaP , iPhiM3);
      const l1t::CaloCluster& clusterSSSW = l1t::CaloTools::getCluster(clusters, iEtaM , iPhiP3);
      const l1t::CaloCluster& clusterSSS  = l1t::CaloTools::getCluster(clusters, iEta  , iPhiP3);
      const l1t::CaloCluster& clusterSSSE = l1t::CaloTools::getCluster(clusters, iEtaP , iPhiP3);
      const l1t::CaloCluster& clusterNNNN = l1t::CaloTools::getCluster(clusters, iEta  , iPhiM4);
      const l1t::CaloCluster& clusterSSSS = l1t::CaloTools::getCluster(clusters, iEta  , iPhiP4);





      // if iEta>1
      bool filterNNWW = (clusterNNWW.hwPt() >= cluster.hwPt());
      bool filterNNW  = (clusterNNW .hwPt() >= cluster.hwPt());
      bool filterNN   = (clusterNN  .hwPt() >  cluster.hwPt());
      bool filterNNE  = (clusterNNE .hwPt() >  cluster.hwPt());
      bool filterNNEE = (clusterNNEE.hwPt() >  cluster.hwPt());
      bool filterNEE  = (clusterNEE .hwPt() >  cluster.hwPt());
      bool filterEE   = (clusterEE  .hwPt() >  cluster.hwPt());
      bool filterSEE  = (clusterSEE .hwPt() >  cluster.hwPt());
      bool filterSSEE = (clusterSSEE.hwPt() >  cluster.hwPt());
      bool filterSSE  = (clusterSSE .hwPt() >  cluster.hwPt());
      bool filterSS   = (clusterSS  .hwPt() >= cluster.hwPt());
      bool filterSSW  = (clusterSSW .hwPt() >= cluster.hwPt());
      bool filterSSWW = (clusterSSWW.hwPt() >= cluster.hwPt());
      bool filterSWW  = (clusterSWW .hwPt() >= cluster.hwPt());
      bool filterWW   = (clusterWW  .hwPt() >= cluster.hwPt());
      bool filterNWW  = (clusterNWW .hwPt() >= cluster.hwPt());
      bool filterNNNW = (clusterNNNW.hwPt() >= cluster.hwPt());
      bool filterNNN  = (clusterNNN .hwPt() >  cluster.hwPt());
      bool filterNNNE = (clusterNNNE.hwPt() >  cluster.hwPt());
      bool filterSSSW = (clusterSSSW.hwPt() >= cluster.hwPt());
      bool filterSSS  = (clusterSSS .hwPt() >= cluster.hwPt());
      bool filterSSSE = (clusterSSSE.hwPt() >  cluster.hwPt());
      bool filterNNNN = (clusterNNNN.hwPt() >  cluster.hwPt());
      bool filterSSSS = (clusterSSSS.hwPt() >= cluster.hwPt());
      if(iEta<-1){
        filterNNWW = (clusterNNWW.hwPt() >  cluster.hwPt());
        filterNNW  = (clusterNNW .hwPt() >  cluster.hwPt());
        filterNN   = (clusterNN  .hwPt() >= cluster.hwPt());
        filterNNE  = (clusterNNE .hwPt() >= cluster.hwPt());
        filterNNEE = (clusterNNEE.hwPt() >= cluster.hwPt());
        filterNEE  = (clusterNEE .hwPt() >= cluster.hwPt());
        filterEE   = (clusterEE  .hwPt() >= cluster.hwPt());
        filterSEE  = (clusterSEE .hwPt() >= cluster.hwPt());
        filterSSEE = (clusterSSEE.hwPt() >= cluster.hwPt());
        filterSSE  = (clusterSSE .hwPt() >= cluster.hwPt());
        filterSS   = (clusterSS  .hwPt() >  cluster.hwPt());
        filterSSW  = (clusterSSW .hwPt() >  cluster.hwPt());
        filterSSWW = (clusterSSWW.hwPt() >  cluster.hwPt());
        filterSWW  = (clusterSWW .hwPt() >  cluster.hwPt());
        filterWW   = (clusterWW  .hwPt() >  cluster.hwPt());
        filterNWW  = (clusterNWW .hwPt() >  cluster.hwPt());
        filterNNNW = (clusterNNNW.hwPt() >  cluster.hwPt());
        filterNNN  = (clusterNNN .hwPt() >= cluster.hwPt());
        filterNNNE = (clusterNNNE.hwPt() >= cluster.hwPt());
        filterSSSW = (clusterSSSW.hwPt() >  cluster.hwPt());
        filterSSS  = (clusterSSS .hwPt() >  cluster.hwPt());
        filterSSSE = (clusterSSSE.hwPt() >= cluster.hwPt());
        filterNNNN = (clusterNNNN.hwPt() >= cluster.hwPt());
        filterSSSS = (clusterSSSS.hwPt() >  cluster.hwPt());
      }
      else if(iEta==1){
        filterNNWW = (clusterNNWW.hwPt() >  cluster.hwPt());
        filterNNW  = (clusterNNW .hwPt() >  cluster.hwPt());
        filterNN   = (clusterNN  .hwPt() >  cluster.hwPt());
        filterNNE  = (clusterNNE .hwPt() >  cluster.hwPt());
        filterNNEE = (clusterNNEE.hwPt() >  cluster.hwPt());
        filterNEE  = (clusterNEE .hwPt() >  cluster.hwPt());
        filterEE   = (clusterEE  .hwPt() >  cluster.hwPt());
        filterSEE  = (clusterSEE .hwPt() >  cluster.hwPt());
        filterSSEE = (clusterSSEE.hwPt() >  cluster.hwPt());
        filterSSE  = (clusterSSE .hwPt() >  cluster.hwPt());
        filterSS   = (clusterSS  .hwPt() >= cluster.hwPt());
        filterSSW  = (clusterSSW .hwPt() >= cluster.hwPt());
        filterSSWW = (clusterSSWW.hwPt() >  cluster.hwPt());
        filterSWW  = (clusterSWW .hwPt() >  cluster.hwPt());
        filterWW   = (clusterWW  .hwPt() >  cluster.hwPt());
        filterNWW  = (clusterNWW .hwPt() >  cluster.hwPt());
        filterNNNW = (clusterNNNW.hwPt() >  cluster.hwPt());
        filterNNN  = (clusterNNN .hwPt() >  cluster.hwPt());
        filterNNNE = (clusterNNNE.hwPt() >  cluster.hwPt());
        filterSSSW = (clusterSSSW.hwPt() >= cluster.hwPt());
        filterSSS  = (clusterSSS .hwPt() >= cluster.hwPt());
        filterSSSE = (clusterSSSE.hwPt() >  cluster.hwPt());
        filterNNNN = (clusterNNNN.hwPt() >  cluster.hwPt());
        filterSSSS = (clusterSSSS.hwPt() >= cluster.hwPt());
      }
      else if(iEta==-1){
        filterNNWW = (clusterNNWW.hwPt() >  cluster.hwPt());
        filterNNW  = (clusterNNW .hwPt() >  cluster.hwPt());
        filterNN   = (clusterNN  .hwPt() >  cluster.hwPt());
        filterNNE  = (clusterNNE .hwPt() >  cluster.hwPt());
        filterNNEE = (clusterNNEE.hwPt() >  cluster.hwPt());
        filterNEE  = (clusterNEE .hwPt() >  cluster.hwPt());
        filterEE   = (clusterEE  .hwPt() >  cluster.hwPt());
        filterSEE  = (clusterSEE .hwPt() >  cluster.hwPt());
        filterSSEE = (clusterSSEE.hwPt() >  cluster.hwPt());
        filterSSE  = (clusterSSE .hwPt() >= cluster.hwPt());
        filterSS   = (clusterSS  .hwPt() >= cluster.hwPt());
        filterSSW  = (clusterSSW .hwPt() >  cluster.hwPt());
        filterSSWW = (clusterSSWW.hwPt() >  cluster.hwPt());
        filterSWW  = (clusterSWW .hwPt() >  cluster.hwPt());
        filterWW   = (clusterWW  .hwPt() >  cluster.hwPt());
        filterNWW  = (clusterNWW .hwPt() >  cluster.hwPt());
        filterNNNW = (clusterNNNW.hwPt() >  cluster.hwPt());
        filterNNN  = (clusterNNN .hwPt() >  cluster.hwPt());
        filterNNNE = (clusterNNNE.hwPt() >  cluster.hwPt());
        filterSSSW = (clusterSSSW.hwPt() >  cluster.hwPt());
        filterSSS  = (clusterSSS .hwPt() >= cluster.hwPt());
        filterSSSE = (clusterSSSE.hwPt() >= cluster.hwPt());
        filterNNNN = (clusterNNNN.hwPt() >  cluster.hwPt());
        filterSSSS = (clusterSSSS.hwPt() >= cluster.hwPt());
      }

      //if(filterNNWW || filterNNW || filterNN || filterWW || filterNWW || filterNNNW)    cluster.setClusterFlag(CaloCluster::INCLUDE_NW, false);
      //if(filterNNW || filterNN || filterNNE || filterNNN)                               cluster.setClusterFlag(CaloCluster::INCLUDE_N , false);
      //if(filterNN || filterNNE || filterNNEE || filterNEE || filterEE || filterNNNE)    cluster.setClusterFlag(CaloCluster::INCLUDE_NE, false);
      //if(filterNEE || filterEE || filterSEE || filterNNE || filterSSE)                  cluster.setClusterFlag(CaloCluster::INCLUDE_E , false);
      //if(filterEE || filterSEE || filterSSEE || filterSSE || filterSS || filterSSSE)    cluster.setClusterFlag(CaloCluster::INCLUDE_SE, false);
      //if(filterSSE || filterSS || filterSSW || filterSSS)                                cluster.setClusterFlag(CaloCluster::INCLUDE_S , false);
      //if(filterSS || filterSSW || filterSSWW || filterSWW || filterWW || filterSSSW)    cluster.setClusterFlag(CaloCluster::INCLUDE_SW, false);
      //if(filterSWW || filterWW || filterNWW || filterNNW || filterSSW)                  cluster.setClusterFlag(CaloCluster::INCLUDE_W , false);
      //if(filterNNW || filterNNE || filterNNNW || filterNNN || filterNNNE || filterNNNN) cluster.setClusterFlag(CaloCluster::INCLUDE_NN, false);
      //if(filterSSW || filterSSE || filterSSSW || filterSSS || filterSSSE || filterSSSS) cluster.setClusterFlag(CaloCluster::INCLUDE_SS, false);
      

      // NOT_IN_FIRMWARE
      const l1t::CaloTower& towerNW  = l1t::CaloTools::getTower(towers, iEtaM , iPhiM);
      const l1t::CaloTower& towerNE  = l1t::CaloTools::getTower(towers, iEtaP , iPhiM);
      const l1t::CaloTower& towerSE  = l1t::CaloTools::getTower(towers, iEtaP , iPhiP);
      const l1t::CaloTower& towerSW  = l1t::CaloTools::getTower(towers, iEtaM , iPhiP);
      const l1t::CaloTower& towerNN  = l1t::CaloTools::getTower(towers, iEta  , iPhiM2);
      const l1t::CaloTower& towerNNW = l1t::CaloTools::getTower(towers, iEtaP , iPhiM2);
      const l1t::CaloTower& towerNNE = l1t::CaloTools::getTower(towers, iEtaM , iPhiM2);
      const l1t::CaloTower& towerSS  = l1t::CaloTools::getTower(towers, iEta  , iPhiP2);
      const l1t::CaloTower& towerSSW = l1t::CaloTools::getTower(towers, iEtaP , iPhiP2);
      const l1t::CaloTower& towerSSE = l1t::CaloTools::getTower(towers, iEtaM , iPhiP2);
      const l1t::CaloTower& towerNNN = l1t::CaloTools::getTower(towers, iEta  , iPhiM3);
      const l1t::CaloTower& towerSSS = l1t::CaloTools::getTower(towers, iEta  , iPhiP3);
      int towerEtNW  = 0;
      int towerEtNE  = 0;
      int towerEtSE  = 0;
      int towerEtSW  = 0;
      int towerEtNN  = 0;
      int towerEtNNE = 0;
      int towerEtNNW = 0;
      int towerEtSS  = 0;
      int towerEtSSE = 0;
      int towerEtSSW = 0;
      int towerEtNNN = 0;
      int towerEtSSS = 0;
      if(m_clusterInput==E){
        towerEtNW  = towerNW .hwEtEm();
        towerEtNE  = towerNE .hwEtEm();
        towerEtSE  = towerSE .hwEtEm();
        towerEtSW  = towerSW .hwEtEm();
        towerEtNN  = towerNN .hwEtEm();
        towerEtNNE = towerNNE.hwEtEm();
        towerEtNNW = towerNNW.hwEtEm();
        towerEtSS  = towerSS .hwEtEm();
        towerEtSSE = towerSSE.hwEtEm();
        towerEtSSW = towerSSW.hwEtEm();
        towerEtNNN = towerNNN.hwEtEm();
        towerEtSSS = towerSSS.hwEtEm();
      }
      else if(m_clusterInput==H){
        towerEtNW  = towerNW .hwEtHad();
        towerEtNE  = towerNE .hwEtHad();
        towerEtSE  = towerSE .hwEtHad();
        towerEtSW  = towerSW .hwEtHad();
        towerEtNN  = towerNN .hwEtHad();
        towerEtNNE = towerNNE.hwEtHad();
        towerEtNNW = towerNNW.hwEtHad();
        towerEtSS  = towerSS .hwEtHad();
        towerEtSSE = towerSSE.hwEtHad();
        towerEtSSW = towerSSW.hwEtHad();
        towerEtNNN = towerNNN.hwEtHad();
        towerEtSSS = towerSSS.hwEtHad();
      }
      else if(m_clusterInput==EH){
        towerEtNW  = towerNW .hwEtEm() + towerNW .hwEtHad();
        towerEtNE  = towerNE .hwEtEm() + towerNE .hwEtHad();
        towerEtSE  = towerSE .hwEtEm() + towerSE .hwEtHad();
        towerEtSW  = towerSW .hwEtEm() + towerSW .hwEtHad();
        towerEtNN  = towerNN .hwEtEm() + towerNN .hwEtHad();
        towerEtNNE = towerNNE.hwEtEm() + towerNNE.hwEtHad();
        towerEtNNW = towerNNW.hwEtEm() + towerNNW.hwEtHad();
        towerEtSS  = towerSS .hwEtEm() + towerSS .hwEtHad();
        towerEtSSE = towerSSE.hwEtEm() + towerSSE.hwEtHad();
        towerEtSSW = towerSSW.hwEtEm() + towerSSW.hwEtHad();
        towerEtNNN = towerNNN.hwEtEm() + towerNNN.hwEtHad();
        towerEtSSS = towerSSS.hwEtEm() + towerSSS.hwEtHad();
      }

      if(filterNNWW || filterNNW || filterNN || filterWW || filterNWW || (filterNNNW && towerEtNNW>=m_clusterThreshold)){
        cluster.setClusterFlag(CaloCluster::INCLUDE_NW, false);
      }
      if(filterNNW || filterNN || filterNNE || (filterNNN && towerEtNN>=m_clusterThreshold)){
        cluster.setClusterFlag(CaloCluster::INCLUDE_N , false);
      }
      if(filterNN || filterNNE || filterNNEE || filterNEE || filterEE || (filterNNNE && towerEtNNE>=m_clusterThreshold)){
        cluster.setClusterFlag(CaloCluster::INCLUDE_NE, false);
      }
      if(filterNEE || filterEE || filterSEE || (filterNNE && towerEtNE>=m_clusterThreshold) || (filterSSE && towerEtSE>=m_clusterThreshold)){
        cluster.setClusterFlag(CaloCluster::INCLUDE_E , false);
      }
      if(filterEE || filterSEE || filterSSEE || filterSSE || filterSS || (filterSSSE && towerEtSSE>=m_clusterThreshold)){
        cluster.setClusterFlag(CaloCluster::INCLUDE_SE, false);
      }
      if(filterSSE || filterSS || filterSSW || (filterSSS && towerEtSS>=m_clusterThreshold)){
        cluster.setClusterFlag(CaloCluster::INCLUDE_S , false);
      }
      if(filterSS || filterSSW || filterSSWW || filterSWW || filterWW || (filterSSSW && towerEtSSW>=m_clusterThreshold)){
        cluster.setClusterFlag(CaloCluster::INCLUDE_SW, false);
      }
      if(filterSWW || filterWW || filterNWW || (filterNNW && towerEtNW>=m_clusterThreshold) || (filterSSW && towerEtSW>=m_clusterThreshold)){
        cluster.setClusterFlag(CaloCluster::INCLUDE_W , false);
      }
      if(filterNNW || filterNNE || filterNNNW || filterNNN || filterNNNE || (filterNNNN && towerEtNNN>=m_clusterThreshold)){
        cluster.setClusterFlag(CaloCluster::INCLUDE_NN, false);
      }
      if(filterSSW || filterSSE || filterSSSW || filterSSS || filterSSSE || (filterSSSS && towerEtSSS>=m_clusterThreshold)){
        cluster.setClusterFlag(CaloCluster::INCLUDE_SS, false);
      }
      // END NOT_IN_FIRMWARE

    }
  }
}

void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::refining(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters){
  // navigator
  l1t::CaloStage2Nav caloNav;

  // trim cluster
  for(size_t clusterNr=0;clusterNr<clusters.size();clusterNr++){
    l1t::CaloCluster& cluster = clusters[clusterNr];
    if( cluster.isValid() ){
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
      if(m_clusterInput==E){
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
      else if(m_clusterInput==H){
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
      else if(m_clusterInput==EH){
        towerEtNW = towerNW.hwEtEm() + towerNW.hwEtHad();
        towerEtN  = towerN .hwEtEm() + towerN .hwEtHad();
        towerEtNE = towerNE.hwEtEm() + towerNE.hwEtHad();
        towerEtE  = towerE .hwEtEm() + towerE .hwEtHad();
        towerEtSE = towerSE.hwEtEm() + towerSE.hwEtHad();
        towerEtS  = towerS .hwEtEm() + towerS .hwEtHad();
        towerEtSW = towerSW.hwEtEm() + towerSW.hwEtHad();
        towerEtW  = towerW .hwEtEm() + towerW .hwEtHad();
        towerEtNN = towerNN.hwEtEm() + towerNN.hwEtHad();
        towerEtSS = towerSS.hwEtEm() + towerSS.hwEtHad();
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

      // trim corners
      if(m_trimCorners) {
        //if(towerEtN<m_clusterThreshold && towerEtW<m_clusterThreshold) cluster.setClusterFlag(CaloCluster::INCLUDE_NW, false);
        //if(towerEtN<m_clusterThreshold && towerEtE<m_clusterThreshold) cluster.setClusterFlag(CaloCluster::INCLUDE_NE, false);
        //if(towerEtS<m_clusterThreshold && towerEtW<m_clusterThreshold) cluster.setClusterFlag(CaloCluster::INCLUDE_SW, false);
        //if(towerEtS<m_clusterThreshold && towerEtE<m_clusterThreshold) cluster.setClusterFlag(CaloCluster::INCLUDE_SE, false);
        // NOT_IN_FIRMWARE
        if(!cluster.checkClusterFlag(CaloCluster::INCLUDE_N) && !cluster.checkClusterFlag(CaloCluster::INCLUDE_W)) cluster.setClusterFlag(CaloCluster::INCLUDE_NW, false);
        if(!cluster.checkClusterFlag(CaloCluster::INCLUDE_N) && !cluster.checkClusterFlag(CaloCluster::INCLUDE_E)) cluster.setClusterFlag(CaloCluster::INCLUDE_NE, false);
        if(!cluster.checkClusterFlag(CaloCluster::INCLUDE_S) && !cluster.checkClusterFlag(CaloCluster::INCLUDE_W)) cluster.setClusterFlag(CaloCluster::INCLUDE_SW, false);
        if(!cluster.checkClusterFlag(CaloCluster::INCLUDE_S) && !cluster.checkClusterFlag(CaloCluster::INCLUDE_E)) cluster.setClusterFlag(CaloCluster::INCLUDE_SE, false);
        // END NOT_IN_FIRMWARE
      }

      // trim one eta-side
      int EtEtaRight = 0;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NE)) EtEtaRight += towerEtNE;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_E))  EtEtaRight += towerEtE;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SE)) EtEtaRight += towerEtSE;
      int EtEtaLeft  = 0;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_NW)) EtEtaLeft += towerEtNW;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_W))  EtEtaLeft += towerEtW;
      if(cluster.checkClusterFlag(CaloCluster::INCLUDE_SW)) EtEtaLeft += towerEtSW;

      // favour most central part
      if(iEta>0) cluster.setClusterFlag(CaloCluster::TRIM_LEFT, (EtEtaRight>EtEtaLeft) );
      else       cluster.setClusterFlag(CaloCluster::TRIM_LEFT, (EtEtaRight>=EtEtaLeft) );

      if(cluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
        cluster.setClusterFlag(CaloCluster::INCLUDE_NW, false);
        cluster.setClusterFlag(CaloCluster::INCLUDE_W , false);
        cluster.setClusterFlag(CaloCluster::INCLUDE_SW, false);
      }
      else{
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


      // Compute fine-grain position
      int fgEta = 0;
      int fgPhi = 0;
      if(EtEtaRight!=0 || EtEtaLeft!=0){
        if(cluster.checkClusterFlag(CaloCluster::TRIM_LEFT)) fgEta = 2;
        else fgEta = 1;
      }
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

void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::trimCorners(bool trimCorners) {
  m_trimCorners = trimCorners;
}
