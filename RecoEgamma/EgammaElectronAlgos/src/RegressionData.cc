#include "RecoEgamma/EgammaElectronAlgos/interface/RegressionData.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"


float RegressionData::seedLeftRightAsym()const
{
  float eLeftRightSum = eLeft()+eRight();
  float eLeftRightDiff = eLeft()-eRight();
  return eLeftRightSum !=0 ? eLeftRightDiff/eLeftRightSum : 0.;
}

float RegressionData::seedTopBottomAsym()const
{
  float eTopBottomSum = eTop()+eBottom();
  float eTopBottomDiff = eTop()-eBottom();
  return eTopBottomSum !=0 ? eTopBottomDiff/eTopBottomSum : 0.;
}
  
float RegressionData::subClusRawEnergy(size_t clusNr)const
{
  if(clusNr<subClusRawEnergy_.size()) return subClusRawEnergy_[clusNr];
  else return 0.;
}

float RegressionData::subClusDEta(size_t clusNr)const
{
  if(clusNr<subClusDEta_.size()) return subClusDEta_[clusNr];
  else return 0.;
}

float RegressionData::subClusDPhi(size_t clusNr)const
{
  if(clusNr<subClusDPhi_.size()) return subClusDPhi_[clusNr];
  else return 0.;
}

void RegressionData::fill(const reco::SuperCluster& superClus,
			  const EcalRecHitCollection* ebRecHits,const EcalRecHitCollection* eeRecHits,
			  const CaloGeometry* geom,const CaloTopology* topology,
			  const reco::VertexCollection* vertices)
{
  clear();
  
  isEB_ = superClus.seed()->hitsAndFractions().at(0).first.subdetId()==EcalBarrel;
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
  sigmaIEtaIPhi_ = std::isnan(localCovs[1]) ? 0. : std::sqrt(localCovs[1]);
  sigmaIPhiIPhi_ = std::isnan(localCovs[2]) ? 0. : std::sqrt(localCovs[2]);
  


  EcalClusterLocal ecalClusterLocal; //not clear why this doesnt use static functions
  float thetaTilt=0,phiTilt=0; //dummy variables that are not used but are required by below function
  void (EcalClusterLocal::*localCoordFunc)(const reco::CaloCluster &, const CaloGeometry &,
  					   float &, float &, int &, int &, 
  					   float &, float &)const;
  localCoordFunc = &EcalClusterLocal::localCoordsEB;
  if(isEB()) localCoordFunc = &EcalClusterLocal::localCoordsEE;
  (ecalClusterLocal.*localCoordFunc)(*superClus.seed(),*geom,
				     seedCrysEtaOrX_,seedCrysPhiOrY_,
				     seedCrysIEtaOrIX_,seedCrysIPhiOrIY_,
				     thetaTilt,phiTilt);

  for( auto clus = superClus.clustersBegin()+1;clus != superClus.clustersEnd(); ++clus ) {
    const float dEta = (*clus)->eta() - superClus.seed()->eta();
    const float dPhi = reco::deltaPhi((*clus)->phi(),superClus.seed()->phi());
    const float dR = std::hypot(dEta,dPhi);
    if(dR > maxSubClusDR_ || maxSubClusDR_ == 999.0f) {
      maxSubClusDR_ = dR;
      maxSubClusDRDEta_ = dEta;
      maxSubClusDRDPhi_ = dPhi;
      maxSubClusDRRawEnergy_ = (*clus)->energy();
    }
    subClusRawEnergy_.push_back((*clus)->energy());
    subClusDEta_.push_back(dEta);
    subClusDPhi_.push_back(dPhi);
    
  }
  
  nrVtx_ = vertices->size();

}

void RegressionData::clear()
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
  
  maxSubClusDR_=0.;
  maxSubClusDRDPhi_=0.;
  maxSubClusDRDEta_=0.;
  maxSubClusDRRawEnergy_=0.;
  
  subClusRawEnergy_.clear();
  subClusDPhi_.clear();
  subClusDEta_.clear();
  
  nrVtx_=0;
  
}
