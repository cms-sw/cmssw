#include "RecoEgamma/EgammaTools/interface/EcalRegressionData.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

float EcalRegressionData::seedLeftRightAsym()const
{
  float eLeftRightSum = eLeft()+eRight();
  float eLeftRightDiff = eLeft()-eRight();
  return eLeftRightSum !=0 ? eLeftRightDiff/eLeftRightSum : 0.;
}

float EcalRegressionData::seedTopBottomAsym()const
{
  float eTopBottomSum = eTop()+eBottom();
  float eTopBottomDiff = eTop()-eBottom();
  return eTopBottomSum !=0 ? eTopBottomDiff/eTopBottomSum : 0.;
}
  
float EcalRegressionData::subClusRawEnergy(size_t clusNr)const
{
  if(clusNr<subClusRawEnergy_.size()) return subClusRawEnergy_[clusNr];
  else return 0.;
}

float EcalRegressionData::subClusDEta(size_t clusNr)const
{
  if(clusNr<subClusDEta_.size()) return subClusDEta_[clusNr];
  else return 0.;
}

float EcalRegressionData::subClusDPhi(size_t clusNr)const
{
  if(clusNr<subClusDPhi_.size()) return subClusDPhi_[clusNr];
  else return 0.;
}

void EcalRegressionData::fill(const reco::SuperCluster& superClus,
			  const EcalRecHitCollection* ebRecHits,const EcalRecHitCollection* eeRecHits,
			  const CaloGeometry* geom,const CaloTopology* topology,
			  int nrVertices)
{
  clear();
  
  const DetId& seedid = superClus.seed()->hitsAndFractions().at(0).first;
  isEB_ = ( seedid.subdetId()==EcalBarrel );
  
  // skip HGCal
  if( seedid.det() == DetId::Forward ) return;
  
  const EcalRecHitCollection* recHits = isEB_ ? ebRecHits : eeRecHits;

  scRawEnergy_ = superClus.rawEnergy();
  scCalibEnergy_ = superClus.correctedEnergy();
  scPreShowerEnergy_ =  superClus.preshowerEnergy();
  scEta_ = superClus.eta();
  scPhi_ = superClus.phi();
  scEtaWidth_ = superClus.etaWidth();
  scPhiWidth_ = superClus.phiWidth();
  scNrAdditionalClusters_ = superClus.clustersSize()-1;

  seedClusEnergy_ = superClus.seed()->energy();
  eMax_ = EcalClusterTools::eMax(*superClus.seed(),recHits);
  e2nd_ = EcalClusterTools::e2nd(*superClus.seed(),recHits);
  e3x3_ = EcalClusterTools::e3x3(*superClus.seed(),recHits,topology);
  eTop_ = EcalClusterTools::eTop(*superClus.seed(),recHits,topology);
  eBottom_ = EcalClusterTools::eBottom(*superClus.seed(),recHits,topology);
  eLeft_ = EcalClusterTools::eLeft(*superClus.seed(),recHits,topology);
  eRight_ = EcalClusterTools::eRight(*superClus.seed(),recHits,topology);
  std::vector<float> localCovs = EcalClusterTools::localCovariances(*superClus.seed(),recHits,topology);
  sigmaIEtaIEta_ = std::isnan(localCovs[0]) ? 0. : std::sqrt(localCovs[0]);
  sigmaIPhiIPhi_ = std::isnan(localCovs[2]) ? 0. : std::sqrt(localCovs[2]);
  
  if(sigmaIEtaIEta_*sigmaIPhiIPhi_>0) sigmaIEtaIPhi_ = localCovs[1]/(sigmaIEtaIEta_*sigmaIPhiIPhi_);
  else if(localCovs[1]>0) sigmaIEtaIPhi_ = 1.;
  else sigmaIEtaIPhi_ = -1.;
  

  EcalClusterLocal ecalClusterLocal; //not clear why this doesnt use static functions
  float thetaTilt=0,phiTilt=0; //dummy variables that are not used but are required by below function
  void (EcalClusterLocal::*localCoordFunc)(const reco::CaloCluster &, const CaloGeometry &,
  					   float &, float &, int &, int &, 
  					   float &, float &)const;
  localCoordFunc = &EcalClusterLocal::localCoordsEB;
  if(!isEB()) localCoordFunc = &EcalClusterLocal::localCoordsEE;
  (ecalClusterLocal.*localCoordFunc)(*superClus.seed(),*geom,
				     seedCrysEtaOrX_,seedCrysPhiOrY_,
				     seedCrysIEtaOrIX_,seedCrysIPhiOrIY_,
				     thetaTilt,phiTilt);

  for( auto clus = superClus.clustersBegin()+1;clus != superClus.clustersEnd(); ++clus ) {
    const float dEta = (*clus)->eta() - superClus.seed()->eta();
    const float dPhi = reco::deltaPhi((*clus)->phi(),superClus.seed()->phi());
    const float dR2 = dEta*dEta+dPhi*dPhi;
    if(dR2 > maxSubClusDR2_ || maxSubClusDR2_ == 998001.) {
      maxSubClusDR2_ = dR2;
      maxSubClusDRDEta_ = dEta;
      maxSubClusDRDPhi_ = dPhi;
      maxSubClusDRRawEnergy_ = (*clus)->energy();
    }
    subClusRawEnergy_.push_back((*clus)->energy());
    subClusDEta_.push_back(dEta);
    subClusDPhi_.push_back(dPhi);
    
  }
  
  nrVtx_ = nrVertices;

}

void EcalRegressionData::clear()
{
  isEB_=false;
  scRawEnergy_=0.;
  scCalibEnergy_=0.;
  scPreShowerEnergy_=0.;
  scEta_=0.;
  scPhi_=0.;
  scEtaWidth_=0.;
  scPhiWidth_=0.;
  scNrAdditionalClusters_=0;
  
  seedClusEnergy_=0.;
  eMax_=0.;
  e2nd_=0.;
  e3x3_=0.;
  eTop_=0.;
  eBottom_=0.;
  eLeft_=0.;
  eRight_=0.;
  sigmaIEtaIEta_=0.;
  sigmaIEtaIPhi_=0.;
  sigmaIPhiIPhi_=0.;
  
  seedCrysPhiOrY_=0.;
  seedCrysEtaOrX_=0.;
  seedCrysIEtaOrIX_=0;
  seedCrysIPhiOrIY_=0;
  
  maxSubClusDR2_=998001.;
  maxSubClusDRDPhi_=999.;
  maxSubClusDRDEta_=999;
  maxSubClusDRRawEnergy_=0.;
  
  subClusRawEnergy_.clear();
  subClusDPhi_.clear();
  subClusDEta_.clear();
  
  nrVtx_=0;
  
}

void EcalRegressionData::fillVec(std::vector<float>& inputVec)const
{
  if(isEB()) fillVecEB_(inputVec);
  else fillVecEE_(inputVec);
}

void EcalRegressionData::fillVecEB_(std::vector<float>& inputVec)const
{
  inputVec.clear();
  inputVec.resize(33);
  inputVec[0] = nrVtx(); //nVtx
  inputVec[1] = scEta(); //scEta
  inputVec[2] = scPhi(); //scPhi
  inputVec[3] = scEtaWidth(); //scEtaWidth
  inputVec[4] = scPhiWidth(); //scPhiWidth
  inputVec[5] = scSeedR9(); //scSeedR9
  inputVec[6] = seedClusEnergyOverSCRawEnergy(); //scSeedRawEnergy/scRawEnergy
  inputVec[7] = eMaxOverSCRawEnergy(); //scSeedEmax/scRawEnergy
  inputVec[8] = e2ndOverSCRawEnergy(); //scSeedE2nd/scRawEnergy
  inputVec[9] = seedLeftRightAsym();//scSeedLeftRightAsym
  inputVec[10] = seedTopBottomAsym();//scSeedTopBottomAsym
  inputVec[11] = sigmaIEtaIEta(); //scSeedSigmaIetaIeta
  inputVec[12] = sigmaIEtaIPhi(); //scSeedSigmaIetaIphi
  inputVec[13] = sigmaIPhiIPhi(); //scSeedSigmaIphiIphi
  inputVec[14] = scNrAdditionalClusters(); //N_ECALClusters
  inputVec[15] = maxSubClusDR(); //clusterMaxDR
  inputVec[16] = maxSubClusDRDPhi(); //clusterMaxDRDPhi
  inputVec[17] = maxSubClusDRDEta(); //clusterMaxDRDEta
  inputVec[18] = maxSubClusDRRawEnergyOverSCRawEnergy(); //clusMaxDRRawEnergy/scRawEnergy
  inputVec[19] = subClusRawEnergyOverSCRawEnergy(SubClusNr::C1); //clusterRawEnergy[0]/scRawEnergy
  inputVec[20] = subClusRawEnergyOverSCRawEnergy(SubClusNr::C2); //clusterRawEnergy[1]/scRawEnergy  float scPreShowerEnergy()const{return scPreShowerEnergy_;}

  inputVec[21] = subClusRawEnergyOverSCRawEnergy(SubClusNr::C3); //clusterRawEnergy[2]/scRawEnergy
  inputVec[22] = subClusDPhi(SubClusNr::C1); //clusterDPhiToSeed[0]
  inputVec[23] = subClusDPhi(SubClusNr::C2); //clusterDPhiToSeed[1]
  inputVec[24] = subClusDPhi(SubClusNr::C3); //clusterDPhiToSeed[2]
  inputVec[25] = subClusDEta(SubClusNr::C1); //clusterDEtaToSeed[0]
  inputVec[26] = subClusDEta(SubClusNr::C2); //clusterDEtaToSeed[1]
  inputVec[27] = subClusDEta(SubClusNr::C3); //clusterDEtaToSeed[2]
  inputVec[28] = seedCrysEtaOrX(); //scSeedCryEta
  inputVec[29] = seedCrysPhiOrY(); //scSeedCryPhi
  inputVec[30] = seedCrysIEtaOrIX(); //scSeedCryIeta
  inputVec[31] = seedCrysIPhiOrIY(); //scSeedCryIphi
  inputVec[32] = scCalibEnergy(); //scCalibratedEnergy
}

void EcalRegressionData::fillVecEE_(std::vector<float>& inputVec)const
{
  inputVec.clear();
  inputVec.resize(33); //this emulates the old behaviour of RegressionHelper, even if past 29 we dont use elements
  inputVec[0] = nrVtx(); //nVtx
  inputVec[1] = scEta(); //scEta
  inputVec[2] = scPhi(); //scPhi
  inputVec[3] = scEtaWidth(); //scEtaWidth
  inputVec[4] = scPhiWidth(); //scPhiWidth
  inputVec[5] = scSeedR9(); //scSeedR9
  inputVec[6] = seedClusEnergyOverSCRawEnergy(); //scSeedRawEnergy/scRawEnergy
  inputVec[7] = eMaxOverSCRawEnergy(); //scSeedEmax/scRawEnergy
  inputVec[8] = e2ndOverSCRawEnergy(); //scSeedE2nd/scRawEnergy
  inputVec[9] = seedLeftRightAsym();//scSeedLeftRightAsym
  inputVec[10] = seedTopBottomAsym();//scSeedTopBottomAsym
  inputVec[11] = sigmaIEtaIEta(); //scSeedSigmaIetaIeta
  inputVec[12] = sigmaIEtaIPhi(); //scSeedSigmaIetaIphi
  inputVec[13] = sigmaIPhiIPhi(); //scSeedSigmaIphiIphi
  inputVec[14] = scNrAdditionalClusters(); //N_ECALClusters
  inputVec[15] = maxSubClusDR(); //clusterMaxDR
  inputVec[16] = maxSubClusDRDPhi(); //clusterMaxDRDPhi
  inputVec[17] = maxSubClusDRDEta(); //clusterMaxDRDEta
  inputVec[18] = maxSubClusDRRawEnergyOverSCRawEnergy(); //clusMaxDRRawEnergy/scRawEnergy
  inputVec[19] = subClusRawEnergyOverSCRawEnergy(SubClusNr::C1); //clusterRawEnergy[0]/scRawEnergy
  inputVec[20] = subClusRawEnergyOverSCRawEnergy(SubClusNr::C2); //clusterRawEnergy[1]/scRawEnergy
  inputVec[21] = subClusRawEnergyOverSCRawEnergy(SubClusNr::C3); //clusterRawEnergy[2]/scRawEnergy
  inputVec[22] = subClusDPhi(SubClusNr::C1); //clusterDPhiToSeed[0]
  inputVec[23] = subClusDPhi(SubClusNr::C2); //clusterDPhiToSeed[1]
  inputVec[24] = subClusDPhi(SubClusNr::C3); //clusterDPhiToSeed[2]
  inputVec[25] = subClusDEta(SubClusNr::C1); //clusterDEtaToSeed[0]
  inputVec[26] = subClusDEta(SubClusNr::C2); //clusterDEtaToSeed[1]
  inputVec[27] = subClusDEta(SubClusNr::C3); //clusterDEtaToSeed[2]
  inputVec[28] = scPreShowerEnergyOverSCRawEnergy();
  inputVec[29] = scCalibEnergy(); //scCalibratedEnergy
}
