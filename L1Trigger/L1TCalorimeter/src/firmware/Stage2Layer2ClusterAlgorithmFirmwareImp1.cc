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
    if(hwEt>=m_seedThreshold)
    {
      math::XYZTLorentzVector emptyP4;
      clusters.push_back( CaloCluster(emptyP4, hwEt, iEta, iPhi) );
      clusters.back().setClusterFlag(CaloCluster::PASS_THRES_SEED);
      clusters.back().setHwSeedPt(hwEt);
      // H/E of the cluster is H/E of the seed
      int hwEtHad = (towers[towerNr].hwEtHad()>=m_hcalThreshold ? towers[towerNr].hwEtHad() : 0);
      int hOverE = (towers[towerNr].hwEtEm()>0 ? (hwEtHad<<8)/towers[towerNr].hwEtEm() : 255);
      if(hOverE>255) hOverE = 255; // bound H/E at 1-? In the future it will be useful to replace with H/(E+H) (or add an other variable), for taus.
      clusters.back().setHOverE(hOverE);
      // FG of the cluster is FG of the seed
      bool fg = (towers[towerNr].hwQual() & (0x1<<2));
      clusters.back().setFgECAL((int)fg);
    }
  }
  // Filter seed: keep only local maxima
  for(size_t clusterNr=0;clusterNr<clusters.size();clusterNr++){
    l1t::CaloCluster& cluster = clusters[clusterNr];
    int iEta = cluster.hwEta();
    int iPhi = cluster.hwPhi();
    int iEtaP = caloNav.offsetIEta(iEta, 1);
    int iEtaM = caloNav.offsetIEta(iEta, -1);
    int iPhiP = caloNav.offsetIPhi(iPhi, 1);
    int iPhiM = caloNav.offsetIPhi(iPhi, -1);
    const l1t::CaloCluster& clusterNW = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiM);
    const l1t::CaloCluster& clusterN  = l1t::CaloTools::getCluster(clusters, iEta , iPhiM);
    const l1t::CaloCluster& clusterNE = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiM);
    const l1t::CaloCluster& clusterE  = l1t::CaloTools::getCluster(clusters, iEtaP, iPhi );
    const l1t::CaloCluster& clusterSE = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiP);
    const l1t::CaloCluster& clusterS  = l1t::CaloTools::getCluster(clusters, iEta , iPhiP);
    const l1t::CaloCluster& clusterSW = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiP);
    const l1t::CaloCluster& clusterW  = l1t::CaloTools::getCluster(clusters, iEtaM, iPhi );
    cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, true);
    if(clusterNW.hwPt() > cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
    if(clusterN .hwPt() > cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
    if(clusterNE.hwPt() > cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
    if(clusterE .hwPt() > cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
    if(clusterSE.hwPt() > cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
    if(clusterS .hwPt() > cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
    if(clusterSW.hwPt() > cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
    if(clusterW .hwPt() > cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
  }

  // add neighbor towers to the seed
  for(size_t clusterNr=0;clusterNr<clusters.size();clusterNr++){
    l1t::CaloCluster& cluster = clusters[clusterNr];
    if( cluster.checkClusterFlag(CaloCluster::PASS_THRES_SEED) ){
      int iEta = cluster.hwEta();
      int iPhi = cluster.hwPhi();
      int iEtaP = caloNav.offsetIEta(iEta, 1);
      int iEtaM = caloNav.offsetIEta(iEta, -1);
      int iPhiP = caloNav.offsetIPhi(iPhi, 1);
      int iPhiM = caloNav.offsetIPhi(iPhi, -1);
      const l1t::CaloTower& towerNW = l1t::CaloTools::getTower(towers, iEtaM, iPhiM);
      const l1t::CaloTower& towerN  = l1t::CaloTools::getTower(towers, iEta , iPhiM);
      const l1t::CaloTower& towerNE = l1t::CaloTools::getTower(towers, iEtaP, iPhiM);
      const l1t::CaloTower& towerE  = l1t::CaloTools::getTower(towers, iEtaP, iPhi );
      const l1t::CaloTower& towerSE = l1t::CaloTools::getTower(towers, iEtaP, iPhiP);
      const l1t::CaloTower& towerS  = l1t::CaloTools::getTower(towers, iEta , iPhiP);
      const l1t::CaloTower& towerSW = l1t::CaloTools::getTower(towers, iEtaM, iPhiP);
      const l1t::CaloTower& towerW  = l1t::CaloTools::getTower(towers, iEtaM, iPhi ); 
      int towerEtNW = 0;
      int towerEtN  = 0;
      int towerEtNE = 0;
      int towerEtE  = 0;
      int towerEtSE = 0;
      int towerEtS  = 0;
      int towerEtSW = 0;
      int towerEtW  = 0;
      if(m_clusterInput==E){
        towerEtNW = towerNW.hwEtEm();
        towerEtN  = towerN .hwEtEm();
        towerEtNE = towerNE.hwEtEm();
        towerEtE  = towerE .hwEtEm();
        towerEtSE = towerSE.hwEtEm();
        towerEtS  = towerS .hwEtEm();
        towerEtSW = towerSW.hwEtEm();
        towerEtW  = towerW .hwEtEm();
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
      }
      towerEtNW = (towerEtNW>=m_clusterThreshold ? towerEtNW : 0);
      towerEtN  = (towerEtN >=m_clusterThreshold ? towerEtN  : 0);
      towerEtNE = (towerEtNE>=m_clusterThreshold ? towerEtNE : 0);
      towerEtE  = (towerEtE >=m_clusterThreshold ? towerEtE  : 0);
      towerEtSE = (towerEtSE>=m_clusterThreshold ? towerEtSE : 0);
      towerEtS  = (towerEtS >=m_clusterThreshold ? towerEtS  : 0);
      towerEtSW = (towerEtSW>=m_clusterThreshold ? towerEtSW : 0);
      towerEtW  = (towerEtW >=m_clusterThreshold ? towerEtW  : 0);
      cluster.setHwPt( cluster.hwPt()+
          towerEtNW+
          towerEtN+
          towerEtNE+
          towerEtE+
          towerEtSE+
          towerEtS+
          towerEtSW+
          towerEtW);
    }
  }

}


void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::filtering(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters){
  // navigator
  l1t::CaloStage2Nav caloNav;

  // Filter cluster: keep only local maxima
  // If two neighbor clusters have the same energy, favor the most central one
  for(size_t clusterNr=0;clusterNr<clusters.size();clusterNr++){
    l1t::CaloCluster& cluster = clusters[clusterNr];
    if( cluster.isValid() ){
      int iEta = cluster.hwEta();
      int iPhi = cluster.hwPhi();
      int iEtaP = caloNav.offsetIEta(iEta, 1);
      int iEtaM = caloNav.offsetIEta(iEta, -1);
      int iPhiP = caloNav.offsetIPhi(iPhi, 1);
      int iPhiM = caloNav.offsetIPhi(iPhi, -1);
      const l1t::CaloCluster& clusterNW = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiM);
      const l1t::CaloCluster& clusterN  = l1t::CaloTools::getCluster(clusters, iEta , iPhiM);
      const l1t::CaloCluster& clusterNE = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiM);
      const l1t::CaloCluster& clusterE  = l1t::CaloTools::getCluster(clusters, iEtaP, iPhi );
      const l1t::CaloCluster& clusterSE = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiP);
      const l1t::CaloCluster& clusterS  = l1t::CaloTools::getCluster(clusters, iEta , iPhiP);
      const l1t::CaloCluster& clusterSW = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiP);
      const l1t::CaloCluster& clusterW  = l1t::CaloTools::getCluster(clusters, iEtaM, iPhi );
      if(iEta>1)
      {
        if(clusterNW.hwSeedPt()==cluster.hwSeedPt() && clusterNW.hwPt()>=cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterN .hwSeedPt()==cluster.hwSeedPt() && clusterN .hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterNE.hwSeedPt()==cluster.hwSeedPt() && clusterNE.hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterE .hwSeedPt()==cluster.hwSeedPt() && clusterE .hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterSE.hwSeedPt()==cluster.hwSeedPt() && clusterSE.hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterS .hwSeedPt()==cluster.hwSeedPt() && clusterS .hwPt()>=cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterSW.hwSeedPt()==cluster.hwSeedPt() && clusterSW.hwPt()>=cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterW .hwSeedPt()==cluster.hwSeedPt() && clusterW .hwPt()>=cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
      }
      else if(iEta<0)
      {
        if(clusterNW.hwSeedPt()==cluster.hwSeedPt() && clusterNW.hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterN .hwSeedPt()==cluster.hwSeedPt() && clusterN .hwPt()>=cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterNE.hwSeedPt()==cluster.hwSeedPt() && clusterNE.hwPt()>=cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterE .hwSeedPt()==cluster.hwSeedPt() && clusterE .hwPt()>=cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterSE.hwSeedPt()==cluster.hwSeedPt() && clusterSE.hwPt()>=cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterS .hwSeedPt()==cluster.hwSeedPt() && clusterS .hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterSW.hwSeedPt()==cluster.hwSeedPt() && clusterSW.hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterW .hwSeedPt()==cluster.hwSeedPt() && clusterW .hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
      }
      else // iEta==1
      {
        if(clusterNW.hwSeedPt()==cluster.hwSeedPt() && clusterNW.hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterN .hwSeedPt()==cluster.hwSeedPt() && clusterN .hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterNE.hwSeedPt()==cluster.hwSeedPt() && clusterNE.hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterE .hwSeedPt()==cluster.hwSeedPt() && clusterE .hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterSE.hwSeedPt()==cluster.hwSeedPt() && clusterSE.hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterS .hwSeedPt()==cluster.hwSeedPt() && clusterS .hwPt()>=cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterSW.hwSeedPt()==cluster.hwSeedPt() && clusterSW.hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
        if(clusterW .hwSeedPt()==cluster.hwSeedPt() && clusterW .hwPt()> cluster.hwPt()) cluster.setClusterFlag(CaloCluster::PASS_FILTER_CLUSTER, false);
      }
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
      int iPhiM  = caloNav.offsetIPhi(iPhi, -1);
      int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
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
      const l1t::CaloCluster& clusterNWW  = l1t::CaloTools::getCluster(clusters, iEtaM2, iPhi);


      // if iEta>1
      bool filterNNWW = (clusterNNWW.isValid() && clusterNNWW.hwPt()>=cluster.hwPt());
      bool filterNNW  = (clusterNNW .isValid() && clusterNNW .hwPt()>=cluster.hwPt());
      bool filterNN   = (clusterNN  .isValid() && clusterNN  .hwPt()>cluster.hwPt());
      bool filterNNE  = (clusterNNE .isValid() && clusterNNE .hwPt()>cluster.hwPt());
      bool filterNNEE = (clusterNNEE.isValid() && clusterNNEE.hwPt()>cluster.hwPt());
      bool filterNEE  = (clusterNEE .isValid() && clusterNEE .hwPt()>cluster.hwPt());
      bool filterEE   = (clusterEE  .isValid() && clusterEE  .hwPt()>cluster.hwPt());
      bool filterSEE  = (clusterSEE .isValid() && clusterSEE .hwPt()>cluster.hwPt());
      bool filterSSEE = (clusterSSEE.isValid() && clusterSSEE.hwPt()>cluster.hwPt());
      bool filterSSE  = (clusterSSE .isValid() && clusterSSE .hwPt()>cluster.hwPt());
      bool filterSS   = (clusterSS  .isValid() && clusterSS  .hwPt()>=cluster.hwPt());
      bool filterSSW  = (clusterSSW .isValid() && clusterSSW .hwPt()>=cluster.hwPt());
      bool filterSSWW = (clusterSSWW.isValid() && clusterSSWW.hwPt()>=cluster.hwPt());
      bool filterSWW  = (clusterSWW .isValid() && clusterSWW .hwPt()>=cluster.hwPt());
      bool filterWW   = (clusterWW  .isValid() && clusterWW  .hwPt()>=cluster.hwPt());
      bool filterNWW  = (clusterNWW .isValid() && clusterNWW .hwPt()>=cluster.hwPt());
      if(iEta<-1)
      {
        filterNNWW = (clusterNNWW.isValid() && clusterNNWW.hwPt()>cluster.hwPt());
        filterNNW  = (clusterNNW .isValid() && clusterNNW .hwPt()>cluster.hwPt());
        filterNN   = (clusterNN  .isValid() && clusterNN  .hwPt()>=cluster.hwPt());
        filterNNE  = (clusterNNE .isValid() && clusterNNE .hwPt()>=cluster.hwPt());
        filterNNEE = (clusterNNEE.isValid() && clusterNNEE.hwPt()>=cluster.hwPt());
        filterNEE  = (clusterNEE .isValid() && clusterNEE .hwPt()>=cluster.hwPt());
        filterEE   = (clusterEE  .isValid() && clusterEE  .hwPt()>=cluster.hwPt());
        filterSEE  = (clusterSEE .isValid() && clusterSEE .hwPt()>=cluster.hwPt());
        filterSSEE = (clusterSSEE.isValid() && clusterSSEE.hwPt()>=cluster.hwPt());
        filterSSE  = (clusterSSE .isValid() && clusterSSE .hwPt()>=cluster.hwPt());
        filterSS   = (clusterSS  .isValid() && clusterSS  .hwPt()>cluster.hwPt());
        filterSSW  = (clusterSSW .isValid() && clusterSSW .hwPt()>cluster.hwPt());
        filterSSWW = (clusterSSWW.isValid() && clusterSSWW.hwPt()>cluster.hwPt());
        filterSWW  = (clusterSWW .isValid() && clusterSWW .hwPt()>cluster.hwPt());
        filterWW   = (clusterWW  .isValid() && clusterWW  .hwPt()>cluster.hwPt());
        filterNWW  = (clusterNWW .isValid() && clusterNWW .hwPt()>cluster.hwPt());
      }
      else if(iEta==1)
      {
        filterNNWW = (clusterNNWW.isValid() && clusterNNWW.hwPt()>cluster.hwPt());
        filterNNW  = (clusterNNW .isValid() && clusterNNW .hwPt()>cluster.hwPt());
        filterNN   = (clusterNN  .isValid() && clusterNN  .hwPt()>cluster.hwPt());
        filterNNE  = (clusterNNE .isValid() && clusterNNE .hwPt()>cluster.hwPt());
        filterNNEE = (clusterNNEE.isValid() && clusterNNEE.hwPt()>cluster.hwPt());
        filterNEE  = (clusterNEE .isValid() && clusterNEE .hwPt()>cluster.hwPt());
        filterEE   = (clusterEE  .isValid() && clusterEE  .hwPt()>cluster.hwPt());
        filterSEE  = (clusterSEE .isValid() && clusterSEE .hwPt()>cluster.hwPt());
        filterSSEE = (clusterSSEE.isValid() && clusterSSEE.hwPt()>cluster.hwPt());
        filterSSE  = (clusterSSE .isValid() && clusterSSE .hwPt()>cluster.hwPt());
        filterSS   = (clusterSS  .isValid() && clusterSS  .hwPt()>=cluster.hwPt());
        filterSSW  = (clusterSSW .isValid() && clusterSSW .hwPt()>=cluster.hwPt());
        filterSSWW = (clusterSSWW.isValid() && clusterSSWW.hwPt()>cluster.hwPt());
        filterSWW  = (clusterSWW .isValid() && clusterSWW .hwPt()>cluster.hwPt());
        filterWW   = (clusterWW  .isValid() && clusterWW  .hwPt()>cluster.hwPt());
        filterNWW  = (clusterNWW .isValid() && clusterNWW .hwPt()>cluster.hwPt());
      }
      else if(iEta==-1)
      {
        filterNNWW = (clusterNNWW.isValid() && clusterNNWW.hwPt()>cluster.hwPt());
        filterNNW  = (clusterNNW .isValid() && clusterNNW .hwPt()>cluster.hwPt());
        filterNN   = (clusterNN  .isValid() && clusterNN  .hwPt()>cluster.hwPt());
        filterNNE  = (clusterNNE .isValid() && clusterNNE .hwPt()>cluster.hwPt());
        filterNNEE = (clusterNNEE.isValid() && clusterNNEE.hwPt()>cluster.hwPt());
        filterNEE  = (clusterNEE .isValid() && clusterNEE .hwPt()>cluster.hwPt());
        filterEE   = (clusterEE  .isValid() && clusterEE  .hwPt()>cluster.hwPt());
        filterSEE  = (clusterSEE .isValid() && clusterSEE .hwPt()>cluster.hwPt());
        filterSSEE = (clusterSSEE.isValid() && clusterSSEE.hwPt()>cluster.hwPt());
        filterSSE  = (clusterSSE .isValid() && clusterSSE .hwPt()>=cluster.hwPt());
        filterSS   = (clusterSS  .isValid() && clusterSS  .hwPt()>=cluster.hwPt());
        filterSSW  = (clusterSSW .isValid() && clusterSSW .hwPt()>cluster.hwPt());
        filterSSWW = (clusterSSWW.isValid() && clusterSSWW.hwPt()>cluster.hwPt());
        filterSWW  = (clusterSWW .isValid() && clusterSWW .hwPt()>cluster.hwPt());
        filterWW   = (clusterWW  .isValid() && clusterWW  .hwPt()>cluster.hwPt());
        filterNWW  = (clusterNWW .isValid() && clusterNWW .hwPt()>cluster.hwPt());
      }

      if(filterNNWW || filterNNW || filterNN || filterWW || filterNWW) cluster.setClusterFlag(CaloCluster::TRIM_NW, true);
      if(filterNNW || filterNN || filterNNE)                           cluster.setClusterFlag(CaloCluster::TRIM_N , true);
      if(filterNN || filterNNE || filterNNEE || filterNEE || filterEE) cluster.setClusterFlag(CaloCluster::TRIM_NE, true);
      if(filterNEE || filterEE || filterSEE)                           cluster.setClusterFlag(CaloCluster::TRIM_E , true);
      if(filterEE || filterSEE || filterSSEE || filterSSE || filterSS) cluster.setClusterFlag(CaloCluster::TRIM_SE, true);
      if(filterSSE || filterSS || filterSSW)                           cluster.setClusterFlag(CaloCluster::TRIM_S , true);
      if(filterSS || filterSSW || filterSSWW || filterSWW || filterWW) cluster.setClusterFlag(CaloCluster::TRIM_SW, true);
      if(filterSWW || filterWW || filterNWW)                           cluster.setClusterFlag(CaloCluster::TRIM_W , true);
    }
  }
}

void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::refining(const std::vector<l1t::CaloTower> & towers, std::vector<l1t::CaloCluster> & clusters){
  // navigator
  l1t::CaloStage2Nav caloNav;

  // trim and extend cluster
  for(size_t clusterNr=0;clusterNr<clusters.size();clusterNr++){
    l1t::CaloCluster& cluster = clusters[clusterNr];
    if( cluster.isValid() ){
      int iEta = cluster.hwEta();
      int iPhi = cluster.hwPhi();
      int iEtaP  = caloNav.offsetIEta(iEta, 1);
      int iEtaM  = caloNav.offsetIEta(iEta, -1);
      int iPhiP  = caloNav.offsetIPhi(iPhi, 1);
      int iPhiP2 = caloNav.offsetIPhi(iPhi, 2);
      int iPhiP3 = caloNav.offsetIPhi(iPhi, 3);
      int iPhiM  = caloNav.offsetIPhi(iPhi, -1);
      int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
      int iPhiM3 = caloNav.offsetIPhi(iPhi, -3);
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
      towerEtNW = (towerEtNW>=m_clusterThreshold ? towerEtNW : 0);
      towerEtN  = (towerEtN >=m_clusterThreshold ? towerEtN  : 0);
      towerEtNE = (towerEtNE>=m_clusterThreshold ? towerEtNE : 0);
      towerEtE  = (towerEtE >=m_clusterThreshold ? towerEtE  : 0);
      towerEtSE = (towerEtSE>=m_clusterThreshold ? towerEtSE : 0);
      towerEtS  = (towerEtS >=m_clusterThreshold ? towerEtS  : 0);
      towerEtSW = (towerEtSW>=m_clusterThreshold ? towerEtSW : 0);
      towerEtW  = (towerEtW >=m_clusterThreshold ? towerEtW  : 0);
      towerEtNN = (towerEtNN>=m_clusterThreshold ? towerEtNN : 0);
      towerEtSS = (towerEtSS>=m_clusterThreshold ? towerEtSS : 0);

      // seems useless to trim towers with 0 energy, but these flags are used for e/g identification
      if(towerEtNW==0) cluster.setClusterFlag(CaloCluster::TRIM_NW, true);
      if(towerEtN ==0) cluster.setClusterFlag(CaloCluster::TRIM_N , true);
      if(towerEtNE==0) cluster.setClusterFlag(CaloCluster::TRIM_NE, true);
      if(towerEtE ==0) cluster.setClusterFlag(CaloCluster::TRIM_E , true);
      if(towerEtSE==0) cluster.setClusterFlag(CaloCluster::TRIM_SE, true);
      if(towerEtS ==0) cluster.setClusterFlag(CaloCluster::TRIM_S , true);
      if(towerEtSW==0) cluster.setClusterFlag(CaloCluster::TRIM_SW, true);
      if(towerEtW ==0) cluster.setClusterFlag(CaloCluster::TRIM_W , true);

      // trim corners
      if(m_trimCorners) {
        if(towerEtN==0 && towerEtW==0) cluster.setClusterFlag(CaloCluster::TRIM_NW, true);
        if(towerEtN==0 && towerEtE==0) cluster.setClusterFlag(CaloCluster::TRIM_NE, true);
        if(towerEtS==0 && towerEtW==0) cluster.setClusterFlag(CaloCluster::TRIM_SW, true);
        if(towerEtS==0 && towerEtE==0) cluster.setClusterFlag(CaloCluster::TRIM_SE, true);
      }

      // trim one eta-side
      int EtEtaRight = towerEtNE + towerEtE + towerEtSE;
      int EtEtaLeft  = towerEtNW + towerEtW + towerEtSW;
      if     (towerEtE   > towerEtW) cluster.setClusterFlag(CaloCluster::TRIM_LEFT, true);
      else if(towerEtE   < towerEtW) cluster.setClusterFlag(CaloCluster::TRIM_RIGHT, true);
      else if(EtEtaRight > EtEtaLeft) cluster.setClusterFlag(CaloCluster::TRIM_LEFT, true);
      else if(EtEtaRight < EtEtaLeft) cluster.setClusterFlag(CaloCluster::TRIM_RIGHT, true);
      else if(cluster.hwEta()>0) cluster.setClusterFlag(CaloCluster::TRIM_RIGHT, true);
      else cluster.setClusterFlag(CaloCluster::TRIM_LEFT, true);

      if(cluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
        cluster.setClusterFlag(CaloCluster::TRIM_NW, true);
        cluster.setClusterFlag(CaloCluster::TRIM_W , true);
        cluster.setClusterFlag(CaloCluster::TRIM_SW, true);
      }
      if(cluster.checkClusterFlag(CaloCluster::TRIM_RIGHT)){
        cluster.setClusterFlag(CaloCluster::TRIM_NE, true);
        cluster.setClusterFlag(CaloCluster::TRIM_E , true);
        cluster.setClusterFlag(CaloCluster::TRIM_SE, true);
      }

      // Extend the cluster
      const l1t::CaloCluster& clusterNN   = l1t::CaloTools::getCluster(clusters, iEta , iPhiM2);
      const l1t::CaloCluster& clusterNNW  = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiM2);
      const l1t::CaloCluster& clusterNNNW = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiM3);
      const l1t::CaloCluster& clusterNNN  = l1t::CaloTools::getCluster(clusters, iEta , iPhiM3);
      const l1t::CaloCluster& clusterNNNE = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiM3);
      const l1t::CaloCluster& clusterNNE  = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiM2);
      const l1t::CaloCluster& clusterSS   = l1t::CaloTools::getCluster(clusters, iEta , iPhiP2);
      const l1t::CaloCluster& clusterSSW  = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiP2);
      const l1t::CaloCluster& clusterSSSW = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiP3);
      const l1t::CaloCluster& clusterSSS  = l1t::CaloTools::getCluster(clusters, iEta , iPhiP3);
      const l1t::CaloCluster& clusterSSSE = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiP3);
      const l1t::CaloCluster& clusterSSE  = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiP2);
      if(towerEtN>=m_clusterThreshold && towerEtNN>=m_clusterThreshold && 
          !(clusterNN.isValid() || clusterNNW.isValid() || clusterNNNW.isValid() || clusterNNN.isValid() || clusterNNNE.isValid() || clusterNNE.isValid())
          ){
        cluster.setClusterFlag(CaloCluster::EXT_UP, true);
      }
      if(towerEtS>=m_clusterThreshold && towerEtSS>=m_clusterThreshold && 
          !(clusterSS.isValid() || clusterSSW.isValid() || clusterSSSW.isValid() || clusterSSS.isValid() || clusterSSSE.isValid() || clusterSSE.isValid())
          ){
        cluster.setClusterFlag(CaloCluster::EXT_DOWN, true);
      }

      // Apply trimming + extension
      if(cluster.checkClusterFlag(CaloCluster::TRIM_NW)) cluster.setHwPt(cluster.hwPt() - towerEtNW);
      if(cluster.checkClusterFlag(CaloCluster::TRIM_N))  cluster.setHwPt(cluster.hwPt() - towerEtN);
      if(cluster.checkClusterFlag(CaloCluster::TRIM_NE)) cluster.setHwPt(cluster.hwPt() - towerEtNE);
      if(cluster.checkClusterFlag(CaloCluster::TRIM_E))  cluster.setHwPt(cluster.hwPt() - towerEtE);
      if(cluster.checkClusterFlag(CaloCluster::TRIM_SE)) cluster.setHwPt(cluster.hwPt() - towerEtSE);
      if(cluster.checkClusterFlag(CaloCluster::TRIM_S))  cluster.setHwPt(cluster.hwPt() - towerEtS);
      if(cluster.checkClusterFlag(CaloCluster::TRIM_SW)) cluster.setHwPt(cluster.hwPt() - towerEtSW);
      if(cluster.checkClusterFlag(CaloCluster::TRIM_W))  cluster.setHwPt(cluster.hwPt() - towerEtW);
      if(cluster.checkClusterFlag(CaloCluster::EXT_UP))  cluster.setHwPt(cluster.hwPt() + towerEtNN);
      if(cluster.checkClusterFlag(CaloCluster::EXT_DOWN))cluster.setHwPt(cluster.hwPt() + towerEtSS);

      // Compute fine-grain position
      int fgEta = 0;
      int fgPhi = 0;
      if(EtEtaRight!=0 || EtEtaLeft!=0){
        if(cluster.checkClusterFlag(CaloCluster::TRIM_LEFT)) fgEta = 2;
        else if(cluster.checkClusterFlag(CaloCluster::TRIM_RIGHT)) fgEta = 1;
      }
      int EtUp   = towerEtNE + towerEtN + towerEtNW + towerEtNN;
      int EtDown = towerEtSE + towerEtS + towerEtSW + towerEtSS;
      if(EtUp>EtDown) fgPhi = 2;
      else if(EtDown>EtUp) fgPhi = 1;
      cluster.setFgEta(fgEta);
      cluster.setFgPhi(fgPhi);
    }
  }
}

void l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::trimCorners(bool trimCorners) {
  m_trimCorners = trimCorners;
}
