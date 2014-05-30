///
/// \class l1t::Stage2Layer2TauAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2TauAlgorithmFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"


l1t::Stage2Layer2TauAlgorithmFirmwareImp1::Stage2Layer2TauAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{


}


l1t::Stage2Layer2TauAlgorithmFirmwareImp1::~Stage2Layer2TauAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloCluster> & clusters,
							      std::vector<l1t::Tau> & taus) {

  merging(clusters, taus);
}


void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::merging(const std::vector<l1t::CaloCluster>& clusters, std::vector<l1t::Tau>& taus){
  // navigator
  l1t::CaloStage2Nav caloNav; 

  for ( auto itr = clusters.cbegin(); itr != clusters.end(); ++itr ) {
    if( itr->isValid() ){
      const l1t::CaloCluster& mainCluster = *itr;
      int iEta = mainCluster.hwEta();
      int iPhi = mainCluster.hwPhi();
      int iEtaP = caloNav.offsetIEta(iEta, 1);
      int iEtaM = caloNav.offsetIEta(iEta, -1);
      int iPhiP2 = caloNav.offsetIPhi(iPhi, 2);
      int iPhiP3 = caloNav.offsetIPhi(iPhi, 3);
      int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
      int iPhiM3 = caloNav.offsetIPhi(iPhi, -3);

      const l1t::CaloCluster& clusterN2  = l1t::CaloTools::getCluster(clusters, iEta, iPhiM2);
      const l1t::CaloCluster& clusterN3  = l1t::CaloTools::getCluster(clusters, iEta, iPhiM3);
      const l1t::CaloCluster& clusterN2W = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiM2);
      const l1t::CaloCluster& clusterN2E = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiM2);
      const l1t::CaloCluster& clusterS2  = l1t::CaloTools::getCluster(clusters, iEta, iPhiP2);
      const l1t::CaloCluster& clusterS3  = l1t::CaloTools::getCluster(clusters, iEta, iPhiP3);
      const l1t::CaloCluster& clusterS2W = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiP2);
      const l1t::CaloCluster& clusterS2E = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiP2);

      std::list<l1t::CaloCluster> satellites;
      if(clusterN2 .isValid()) satellites.push_back(clusterN2);
      if(clusterN3 .isValid()) satellites.push_back(clusterN3);
      if(clusterN2W.isValid()) satellites.push_back(clusterN2W);
      if(clusterN2E.isValid()) satellites.push_back(clusterN2E);
      if(clusterS2 .isValid()) satellites.push_back(clusterS2);
      if(clusterS3 .isValid()) satellites.push_back(clusterS3);
      if(clusterS2W.isValid()) satellites.push_back(clusterS2W);
      if(clusterS2E.isValid()) satellites.push_back(clusterS2E);

      if(satellites.size()>0) {
        satellites.sort();
        const l1t::CaloCluster& secondaryCluster = satellites.back();
        if(mainCluster>secondaryCluster) {
          math::XYZTLorentzVector p4;
          l1t::Tau tau( p4, mainCluster.hwPt()+secondaryCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
          taus.push_back(tau);
        }
      }
      else {
        math::XYZTLorentzVector p4;
        l1t::Tau tau( p4, mainCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
        taus.push_back(tau);
      }
    }
  }
}
