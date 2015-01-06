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

  loadCalibrationLuts();
}


l1t::Stage2Layer2TauAlgorithmFirmwareImp1::~Stage2Layer2TauAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloCluster> & clusters,
															 const std::vector<l1t::CaloTower>& towers,
							      							 std::vector<l1t::Tau> & taus) {

  merging(clusters, towers, taus);
  
}


// FIXME: to be organized better
void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::merging(const std::vector<l1t::CaloCluster>& clusters, const std::vector<l1t::CaloTower>& towers, std::vector<l1t::Tau>& taus){
  //std::cout<<"---------------   NEW EVENT -----------------------------\n";
  //std::cout<<"---------------------------------------------------------\n";
  // navigator
  l1t::CaloStage2Nav caloNav; 

  // Temp copy of clusters (needed to set merging flags)
  std::vector<l1t::CaloCluster> tmpClusters(clusters);
  // First loop: setting merging flags
  for ( auto itr = tmpClusters.begin(); itr != tmpClusters.end(); ++itr ) {
    if( itr->isValid() ){
      l1t::CaloCluster& mainCluster = *itr;
      int iEta = mainCluster.hwEta();
      int iPhi = mainCluster.hwPhi();
      int iEtaP = caloNav.offsetIEta(iEta, 1);
      int iEtaM = caloNav.offsetIEta(iEta, -1);
      int iPhiP2 = caloNav.offsetIPhi(iPhi, 2);
      int iPhiP3 = caloNav.offsetIPhi(iPhi, 3);
      int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
      int iPhiM3 = caloNav.offsetIPhi(iPhi, -3);


      const l1t::CaloCluster& clusterN2  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiM2);
      const l1t::CaloCluster& clusterN3  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiM3);
      const l1t::CaloCluster& clusterN2W = l1t::CaloTools::getCluster(tmpClusters, iEtaM, iPhiM2);
      const l1t::CaloCluster& clusterN2E = l1t::CaloTools::getCluster(tmpClusters, iEtaP, iPhiM2);
      const l1t::CaloCluster& clusterS2  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiP2);
      const l1t::CaloCluster& clusterS3  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiP3);
      const l1t::CaloCluster& clusterS2W = l1t::CaloTools::getCluster(tmpClusters, iEtaM, iPhiP2);
      const l1t::CaloCluster& clusterS2E = l1t::CaloTools::getCluster(tmpClusters, iEtaP, iPhiP2);

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

        if(secondaryCluster>mainCluster) {
          // is secondary
          mainCluster.setClusterFlag(CaloCluster::IS_SECONDARY, true);
          // to be merged up or down?
          if(secondaryCluster.hwPhi()==iPhiP2 || secondaryCluster.hwPhi()==iPhiP3) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_UPDOWN, true); // 1 (down)
          }
          else if(secondaryCluster.hwPhi()==iPhiM2 || secondaryCluster.hwPhi()==iPhiM3) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_UPDOWN, false); // 0 (up)
          }
          // to be merged left or right?
          if(secondaryCluster.hwEta()==iEtaP) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_LEFTRIGHT, true); // 1 (right)
          }
          else if(secondaryCluster.hwEta()==iEta || secondaryCluster.hwEta()==iEtaM) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_LEFTRIGHT, false); // 0 (left)
          }
        }
        else {
          // is main cluster
          mainCluster.setClusterFlag(CaloCluster::IS_SECONDARY, false);
          // to be merged up or down?
          if(secondaryCluster.hwPhi()==iPhiP2 || secondaryCluster.hwPhi()==iPhiP3) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_UPDOWN, true); // 1 (down)
          }
          else if(secondaryCluster.hwPhi()==iPhiM2 || secondaryCluster.hwPhi()==iPhiM3) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_UPDOWN, false); // 0 (up)
          }
          // to be merged left or right?
          if(secondaryCluster.hwEta()==iEtaP) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_LEFTRIGHT, true); // 1 (right)
          }
          else if(secondaryCluster.hwEta()==iEta || secondaryCluster.hwEta()==iEtaM) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_LEFTRIGHT, false); // 0 (left)
          }
        }
      }
    }
  }

  // Second loop: do the actual merging based on merging flags
  for ( auto itr = tmpClusters.begin(); itr != tmpClusters.end(); ++itr ) {
    if( itr->isValid() ){
      l1t::CaloCluster& mainCluster = *itr;
      int iEta = mainCluster.hwEta();
      int iPhi = mainCluster.hwPhi();
	  
      // physical eta/phi
      double eta = 0.;
      double phi = 0.;
      double seedEta     = CaloTools::towerEta(mainCluster.hwEta());
      double seedEtaSize = CaloTools::towerEtaSize(mainCluster.hwEta());
      double seedPhi     = CaloTools::towerPhi(mainCluster.hwEta(), mainCluster.hwPhi());
      double seedPhiSize = CaloTools::towerPhiSize(mainCluster.hwEta());
      if     (mainCluster.fgEta()==0) eta = seedEta; // center
      else if(mainCluster.fgEta()==2) eta = seedEta + seedEtaSize*0.25; // center + 1/4
      else if(mainCluster.fgEta()==1) eta = seedEta - seedEtaSize*0.25; // center - 1/4
      if     (mainCluster.fgPhi()==0) phi = seedPhi; // center
      else if(mainCluster.fgPhi()==2) phi = seedPhi + seedPhiSize*0.25; // center + 1/4
      else if(mainCluster.fgPhi()==1) phi = seedPhi - seedPhiSize*0.25; // center - 1/4


      int iEtaP = caloNav.offsetIEta(iEta, 1);
      int iEtaM = caloNav.offsetIEta(iEta, -1);
      int iPhiP = caloNav.offsetIPhi(iPhi, 1);
      int iPhiM = caloNav.offsetIPhi(iPhi, -1);      
      int iPhiP2 = caloNav.offsetIPhi(iPhi, 2);
      int iPhiP3 = caloNav.offsetIPhi(iPhi, 3);
      int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
      int iPhiM3 = caloNav.offsetIPhi(iPhi, -3);

      const l1t::CaloCluster& clusterN2  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiM2);
      const l1t::CaloCluster& clusterN3  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiM3);
      const l1t::CaloCluster& clusterN2W = l1t::CaloTools::getCluster(tmpClusters, iEtaM, iPhiM2);
      const l1t::CaloCluster& clusterN2E = l1t::CaloTools::getCluster(tmpClusters, iEtaP, iPhiM2);
      const l1t::CaloCluster& clusterS2  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiP2);
      const l1t::CaloCluster& clusterS3  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiP3);
      const l1t::CaloCluster& clusterS2W = l1t::CaloTools::getCluster(tmpClusters, iEtaM, iPhiP2);
      const l1t::CaloCluster& clusterS2E = l1t::CaloTools::getCluster(tmpClusters, iEtaP, iPhiP2);

      std::list<l1t::CaloCluster> satellites;
      if(clusterN2 .isValid()) satellites.push_back(clusterN2);
      if(clusterN3 .isValid()) satellites.push_back(clusterN3);
      if(clusterN2W.isValid()) satellites.push_back(clusterN2W);
      if(clusterN2E.isValid()) satellites.push_back(clusterN2E);
      if(clusterS2 .isValid()) satellites.push_back(clusterS2);
      if(clusterS3 .isValid()) satellites.push_back(clusterS3);
      if(clusterS2W.isValid()) satellites.push_back(clusterS2W);
      if(clusterS2E.isValid()) satellites.push_back(clusterS2E);

      // neighbour exists
      if(satellites.size()>0) {
        satellites.sort();
        const l1t::CaloCluster& secondaryCluster = satellites.back();
        // this is the most energetic cluster
        // merge with the secondary cluster if it is not merged to an other one
        if(mainCluster>secondaryCluster) {
          bool canBeMerged = true;
          bool mergeUp = (secondaryCluster.hwPhi()==iPhiM2 || secondaryCluster.hwPhi()==iPhiM3);
          bool mergeLeft = (secondaryCluster.hwEta()==iEtaM);
          bool mergeRight = (secondaryCluster.hwEta()==iEtaP);

          if(mergeUp && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN)) canBeMerged = false;
          if(!mergeUp && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN)) canBeMerged = false;
          if(mergeLeft && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)) canBeMerged = false;
          if(mergeRight && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)) canBeMerged = false;
          if(canBeMerged) {
            double calibPt = calibratedPt(mainCluster.hwPtEm()+secondaryCluster.hwPtEm(), mainCluster.hwPtHad()+secondaryCluster.hwPtHad(), mainCluster.hwEta());
            math::PtEtaPhiMLorentzVector p4(calibPt, eta, phi, 0.);
            l1t::Tau tau( p4, mainCluster.hwPt()+secondaryCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
            taus.push_back(tau);
            
            //std::cout << "===================== IS MERGED ========================" << std::endl;


			int hwFootPrint=0;
            int hwEtSum=0;
            
			//std::cout << "taus.back().pt(): " << taus.back().pt() << std::endl;
			if(mainCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN) && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN)){    

            	if(mainCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT) && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)){
					// SumEt in (ieta,iphi) = 5x9 centered in the cluster.Eta+1, cluster.Phi+1
					hwEtSum = CaloTools::calHwEtSum(iEtaP,iPhiP,towers,-1*params_->tauIsoAreaNrTowersEta(),params_->tauIsoAreaNrTowersEta(),
													-1*params_->tauIsoAreaNrTowersPhi(),params_->tauIsoAreaNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO); 

					if(mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && !secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
            			
                	    //Evaluation of the tau footprint of a size 2x5, the slices in ieta are ieta=cluster.Eta to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiP,towers,0,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(!mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are from ieta=cluster.Eta-1 to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiP,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are from ieta=cluster.Eta to ieta=cluster.Eta+2 
						hwFootPrint = CaloTools::calHwEtSum(iEtaP,iPhiP,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);

                    }
					if(!mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && !secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are from ieta=cluster.Eta-1 to ieta=cluster.Eta+1 
						hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiP,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                	}
                }
				if(!mainCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT) && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)){
					// SumEt in (ieta,iphi) = 5x9 centered in the cluster.Eta-1, cluster.Phi+1
					hwEtSum = CaloTools::calHwEtSum(iEtaM,iPhiP,towers,-1*params_->tauIsoAreaNrTowersEta(),params_->tauIsoAreaNrTowersEta(),
              										-1*params_->tauIsoAreaNrTowersPhi(),params_->tauIsoAreaNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO); 
					if(mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && !secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are from ieta=cluster.Eta-1 to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiP,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(!mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 2x5, the slices in ieta are ieta=cluster.Eta-1 && ieta=cluster.Eta 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiP,towers,-1,0,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are ieta=cluster.Eta-1 to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiP,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(!mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && !secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are ieta=cluster.Eta-2 to ieta=cluster.Eta 
                        hwFootPrint = CaloTools::calHwEtSum(iEtaM,iPhiP,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
                }
				if(!mainCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT) && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)){
					// SumEt in (ieta,iphi) = 5x9 centered in the cluster.Eta, cluster.Phi+1
                    hwEtSum = CaloTools::calHwEtSum(iEta,iPhiP,towers,-1*params_->tauIsoAreaNrTowersEta(),params_->tauIsoAreaNrTowersEta(),
													-1*params_->tauIsoAreaNrTowersPhi(),params_->tauIsoAreaNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO); 
					
                    if(mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && !secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are ieta=cluster.Eta-1 to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiP,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(!mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are ieta=cluster.Eta-1 to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiP,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 2x5, the slices in ieta are ieta=cluster.Eta && ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiP,towers,0,+1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(!mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && !secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 2x5, the slices in ieta are ieta=cluster.Eta-1 && ieta=cluster.Eta 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiP,towers,-1,0,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
                }
            }
			if(!mainCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN) && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN)){   

            	if(mainCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT) && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)){
			    	// SumEt in (ieta,iphi) = 5x9 centered in the cluster.Eta+1, cluster.Phi-1
			    	hwEtSum = CaloTools::calHwEtSum(iEtaM,iPhiP,towers,-1*params_->tauIsoAreaNrTowersEta(),params_->tauIsoAreaNrTowersEta(),
                								-1*params_->tauIsoAreaNrTowersPhi(),params_->tauIsoAreaNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO); 

					if(mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && !secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 2x5, the slices in ieta are ieta=cluster.Eta && ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiM,towers,0,+1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(!mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are ieta=cluster.Eta-1 to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiM,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                     }
					if(mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are ieta=cluster.Eta to ieta=cluster.Eta+2 
                        hwFootPrint = CaloTools::calHwEtSum(iEtaP,iPhiM,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(!mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && !secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are ieta=cluster.Eta-1 to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiM,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                     }
                }
				if(!mainCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT) && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)){
			    	// SumEt in (ieta,iphi) = 5x9 centered in the cluster.Eta-1, cluster.Phi-1
			    	hwEtSum = CaloTools::calHwEtSum(iEtaM,iPhiM,towers,-1*params_->tauIsoAreaNrTowersEta(),params_->tauIsoAreaNrTowersEta(),
                								-1*params_->tauIsoAreaNrTowersPhi(),params_->tauIsoAreaNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO); 

					if(mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && !secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are ieta=cluster.Eta-1 to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiM,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(!mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 2x5, the slices in ieta are ieta=cluster.Eta-1 && ieta=cluster.Eta 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiM,towers,-1,0,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are ieta=cluster.Eta-1 to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiM,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(!mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && !secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are ieta=cluster.Eta-2 to ieta=cluster.Eta 
                        hwFootPrint = CaloTools::calHwEtSum(iEtaM,iPhiM,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
                }
				if(!mainCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT) && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)){
					// SumEt in (ieta,iphi) = 5x9 centered in the cluster.Eta, cluster.Phi-1 or Phi-2
                    hwEtSum = CaloTools::calHwEtSum(iEta,iPhiM,towers,-1*params_->tauIsoAreaNrTowersEta(),params_->tauIsoAreaNrTowersEta(),
              											-1*params_->tauIsoAreaNrTowersPhi(),params_->tauIsoAreaNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO); 

					if(mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && !secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are ieta=cluster.Eta-1 to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiM,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(!mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 3x5, the slices in ieta are ieta=cluster.Eta-1 to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiM,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 2x5, the slices in ieta are ieta=cluster.Eta to ieta=cluster.Eta+1 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiM,towers,0,+1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
					if(!mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT) && !secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT)){
						//Evaluation of the tau footprint of a size 2x5, the slices in ieta are ieta=cluster.Eta-1 to ieta=cluster.Eta 
                        hwFootPrint = CaloTools::calHwEtSum(iEta,iPhiM,towers,-1,0,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                    }
            	}
            }
            int isolBit = 0;
        	
			int nrTowers = CaloTools::calNrTowers(-1*params_->tauPUSParam(1),params_->tauPUSParam(1),1,72,towers,1,999,CaloTools::CALO);
        	unsigned int lutAddress = isoLutIndex(calibPt, nrTowers);
        	
        	isolBit = hwEtSum-hwFootPrint <= (params_->tauIsolationLUT()->data(lutAddress));
        	taus.back().setHwIso(isolBit);

          }
          else {
            double calibPt = calibratedPt(mainCluster.hwPtEm(), mainCluster.hwPtHad(), mainCluster.hwEta());
            math::PtEtaPhiMLorentzVector p4(calibPt, eta, phi, 0.);
            l1t::Tau tau( p4, mainCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
            taus.push_back(tau);
            //std::cout<<"   Make tau, No merging\n";
            
            // Isolation part
            int hwEtSum = CaloTools::calHwEtSum(mainCluster.hwEta(),mainCluster.hwPhi(),towers,-1*params_->tauIsoAreaNrTowersEta(),params_->tauIsoAreaNrTowersEta(),
              									-1*params_->tauIsoAreaNrTowersPhi(),params_->tauIsoAreaNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                                                
            int hwFootPrint = isoCalTauHwFootPrint(mainCluster,towers);
            
            int isolBit = 0;
			int nrTowers = CaloTools::calNrTowers(-1*params_->tauPUSParam(1),params_->tauPUSParam(1),1,72,towers,1,999,CaloTools::CALO);
        	unsigned int lutAddress = isoLutIndex(calibPt, nrTowers);
        	
        	isolBit = hwEtSum-hwFootPrint <= params_->tauIsolationLUT()->data(lutAddress);
			taus.back().setHwIso(isolBit);
          }
        }
        else {
          bool canBeKept = false;
          bool mergeUp = (secondaryCluster.hwPhi()==iPhiM2 || secondaryCluster.hwPhi()==iPhiM3);
          bool mergeLeft = (secondaryCluster.hwEta()==iEtaM);
          bool mergeRight = (secondaryCluster.hwEta()==iEtaP);

          if(mergeUp && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN)) canBeKept = true;
          if(!mergeUp && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN)) canBeKept = true;
          if(mergeLeft && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)) canBeKept = true;
          if(mergeRight && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)) canBeKept = true;
          if(canBeKept) {
            double calibPt = calibratedPt(mainCluster.hwPtEm(), mainCluster.hwPtHad(), mainCluster.hwEta());
            math::PtEtaPhiMLorentzVector p4(calibPt, eta, phi, 0.);
            l1t::Tau tau( p4, mainCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
            taus.push_back(tau);
            //std::cout<<"   Make tau, No merging\n";
            
            // Isolation part
      		int hwEtSum = CaloTools::calHwEtSum(mainCluster.hwEta(),mainCluster.hwPhi(),towers,-1*params_->tauIsoAreaNrTowersEta(),params_->tauIsoAreaNrTowersEta(),
												-1*params_->tauIsoAreaNrTowersPhi(),params_->tauIsoAreaNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                                            
      		int hwFootPrint = isoCalTauHwFootPrint(mainCluster,towers);
            
            int isolBit = 0;
			int nrTowers = CaloTools::calNrTowers(-1*params_->tauPUSParam(1),params_->tauPUSParam(1),1,72,towers,1,999,CaloTools::CALO);
        	unsigned int lutAddress = isoLutIndex(calibPt, nrTowers);
        	
        	isolBit = hwEtSum-hwFootPrint <= params_->tauIsolationLUT()->data(lutAddress);
            taus.back().setHwIso(isolBit);

          }
        }
      }
      else {
        double calibPt = calibratedPt(mainCluster.hwPtEm(), mainCluster.hwPtHad(), mainCluster.hwEta());
        math::PtEtaPhiMLorentzVector p4(calibPt, eta, phi, 0.);
        l1t::Tau tau( p4, mainCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
        taus.push_back(tau);
        //std::cout<<"   Make tau, No merging\n";
        
        // Isolation part
		int hwEtSum = CaloTools::calHwEtSum(mainCluster.hwEta(),mainCluster.hwPhi(),towers,-1*params_->tauIsoAreaNrTowersEta(),params_->tauIsoAreaNrTowersEta(),
        									-1*params_->tauIsoAreaNrTowersPhi(),params_->tauIsoAreaNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
                                        
      	int hwFootPrint = isoCalTauHwFootPrint(mainCluster,towers);

        int isolBit = 0;
		int nrTowers = CaloTools::calNrTowers(-1*params_->tauPUSParam(1),params_->tauPUSParam(1),1,72,towers,1,999,CaloTools::CALO);
        unsigned int lutAddress = isoLutIndex(calibPt, nrTowers);
        
        isolBit = hwEtSum-hwFootPrint <= params_->tauIsolationLUT()->data(lutAddress);
      	taus.back().setHwIso(isolBit);

      }
    }
  }
}


//calculates the footprint of the tau in hardware values
int l1t::Stage2Layer2TauAlgorithmFirmwareImp1::isoCalTauHwFootPrint(const l1t::CaloCluster& clus,const std::vector<l1t::CaloTower>& towers) 
{
  int iEta=clus.hwEta();
  int iPhi=clus.hwPhi();
  int totHwFootPrint = CaloTools::calHwEtSum(iEta,iPhi,towers,-1,1,-1*params_->tauIsoVetoNrTowersPhi(),params_->tauIsoVetoNrTowersPhi(),params_->tauPUSParam(2),CaloTools::CALO);
  return totHwFootPrint;
}




void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::loadCalibrationLuts()
{
  float minScale    = 0.;
  float maxScale    = 2.;
  float minScaleEta = 0.5;
  float maxScaleEta = 1.5;
  offsetBarrelEH_   = 0.5;
  offsetBarrelH_    = 1.5;
  offsetEndcapsEH_  = 0.;
  offsetEndcapsH_   = 1.5;

  // In the combined calibration LUT, upper 3-bits are used as LUT index:
  // (0=BarrelA, 1=BarrelB, 2=BarrelC, 3=EndCapA, 4=EndCapA, 5=EndCapA, 6=Eta)
  enum {LUT_UPPER = 3};
  enum {LUT_OFFSET = 0x80};
  l1t::LUT* lut = params_->tauCalibrationLUT();
  unsigned int size = (1 << lut->nrBitsData());
  unsigned int nBins = (1 << (lut->nrBitsAddress() - LUT_UPPER));


  std::vector<float> emptyCoeff;
  emptyCoeff.resize(nBins,0.);
  float binSize = (maxScale-minScale)/(float)size;
  for(unsigned iLut=0; iLut < 6; ++iLut ) {
    coefficients_.push_back(emptyCoeff);
    for(unsigned addr=0;addr<nBins;addr++) {
      float y = (float)lut->data(iLut*LUT_OFFSET + addr);
      coefficients_[iLut][addr] = minScale + binSize*y;
    }
  }

  size = (1 << lut->nrBitsData());
  nBins = (1 << 6); // can't auto-extract this now due to combined LUT.
  
  emptyCoeff.resize(nBins,0.);
  binSize = (maxScaleEta-minScaleEta)/(float)size;
  coefficients_.push_back(emptyCoeff);
  for(unsigned addr=0;addr<nBins;addr++) {
    float y = (float)lut->data(6*LUT_OFFSET + addr);
    coefficients_.back()[addr] = minScaleEta + binSize*y;
  }

}


double l1t::Stage2Layer2TauAlgorithmFirmwareImp1::calibratedPt(int hwPtEm, int hwPtHad, int ieta)
{
  // ET calibration
  bool barrel = (ieta<=17);
  unsigned int nBins = coefficients_[0].size();
  double e = (double)hwPtEm*params_->tauLsb();
  double h = (double)hwPtHad*params_->tauLsb();
  double calibPt = 0.;
  int ilutOffset = (barrel) ? 0: 3;
  unsigned int ibin=(unsigned int)(floor(e+h));
  if (ibin >= nBins -1) ibin = nBins-1;
  if(e>0.) {
    double offset = (barrel) ? offsetBarrelEH_ : offsetEndcapsEH_;
    calibPt = e*coefficients_[ilutOffset][ibin] + h*coefficients_[1+ilutOffset][ibin] + offset;
  }
  else {
    double offset = (barrel) ? offsetBarrelH_ : offsetEndcapsH_;
    calibPt = h*coefficients_[2+ilutOffset][ibin]+offset;
  } 

  // eta calibration
  if(ieta<-28) ieta=-28;
  if(ieta>28) ieta=28;
  ibin = (ieta>0 ? ieta+27 : ieta+28);
  calibPt *= coefficients_.back()[ibin];

  return calibPt;
}

// LUT FORMAT: N=1024 (10 bit) blocks for each value of nrTowers
// Each one of this blocks has a substructure of N=256 (8 bit) for the energy value
unsigned int l1t::Stage2Layer2TauAlgorithmFirmwareImp1::isoLutIndex(int Et, unsigned int nrTowers)
{
   const unsigned int kTowerGranularity=params_->tauPUSParam(0);
   unsigned int nrTowersNormed = nrTowers/kTowerGranularity;
      
   if (nrTowersNormed > 1023) nrTowersNormed  = 1023; // 10 bits for towers
   int kTowerOffs = 256*nrTowersNormed;
   
   if (Et > 255)  Et = 255; // 8 bit for E
   
   return (kTowerOffs + Et);
}
