#ifndef EgammaElectronAlgos_EcalRegressionData_h
#define EgammaElectronAlgos_EcalRegressionData_h

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include <vector>
#include <cmath>

class CaloGeometry;
class CaloTopology;

namespace reco{
  class SuperCluster;
}

class EcalRegressionData {
public:
  EcalRegressionData(){clear();}
  
  //this exists due to concerns that sub-cluster 1 is actually accessed 
  //by subClusRawE_[0] and could potentially cause bugs 
  //this although slightly wordy, makes it absolutely clear 
  enum class SubClusNr{
    C1=0,
    C2=1,
    C3=2
  };

  //direct accessors  
  bool isEB()const{return isEB_;}
  float scRawEnergy()const{return scRawEnergy_;}
  float scCalibEnergy()const{return scCalibEnergy_;}
  float scPreShowerEnergy()const{return scPreShowerEnergy_;}
  float scEta()const{return scEta_;}
  float scPhi()const{return scPhi_;}
  float scEtaWidth()const{return scEtaWidth_;}
  float scPhiWidth()const{return scPhiWidth_;}
  int scNrAdditionalClusters()const{return scNrAdditionalClusters_;}
  float seedClusEnergy()const{return seedClusEnergy_;}
  float eMax()const{return eMax_;}
  float e2nd()const{return e2nd_;}
  float e3x3()const{return e3x3_;}
  float eTop()const{return eTop_;}
  float eBottom()const{return eBottom_;}
  float eLeft()const{return eLeft_;}
  float eRight()const{return eRight_;}
  float sigmaIEtaIEta()const{return sigmaIEtaIEta_;}
  float sigmaIEtaIPhi()const{return sigmaIEtaIPhi_;}
  float sigmaIPhiIPhi()const{return sigmaIPhiIPhi_;}
 
  float seedCrysPhiOrY()const{return seedCrysPhiOrY_;}
  float seedCrysEtaOrX()const{return seedCrysEtaOrX_;}
  float seedCrysIEtaOrIX()const{return seedCrysIEtaOrIX_;}
  float seedCrysIPhiOrIY()const{return seedCrysIPhiOrIY_;}
  float maxSubClusDR()const{return std::sqrt(maxSubClusDR2_);}
  float maxSubClusDRDPhi()const{return maxSubClusDRDPhi_;}
  float maxSubClusDRDEta()const{return maxSubClusDRDEta_;}
  float maxSubClusDRRawEnergy()const{return maxSubClusDRRawEnergy_;}
  const std::vector<float>& subClusRawEnergy()const{return subClusRawEnergy_;}
  const std::vector<float>& subClusDPhi()const{return subClusDPhi_;}
  const std::vector<float>& subClusDEta()const{return subClusDEta_;}
  int nrVtx()const{return nrVtx_;}

  //indirect accessors
  float scPreShowerEnergyOverSCRawEnergy()const{return divideBySCRawEnergy_(scPreShowerEnergy());}
  float scSeedR9()const{return divideBySCRawEnergy_(e3x3());}
  float seedClusEnergyOverSCRawEnergy()const{return divideBySCRawEnergy_(seedClusEnergy());}
  float eMaxOverSCRawEnergy()const{return divideBySCRawEnergy_(eMax());}
  float e2ndOverSCRawEnergy()const{return divideBySCRawEnergy_(e2nd());}
  float seedLeftRightAsym()const;
  float seedTopBottomAsym()const;  
  float maxSubClusDRRawEnergyOverSCRawEnergy()const{return divideBySCRawEnergy_(maxSubClusDRRawEnergy());}
  float subClusRawEnergyOverSCRawEnergy(size_t clusNr)const{return divideBySCRawEnergy_(subClusRawEnergy(clusNr));}
  float subClusRawEnergy(size_t clusNr)const;
  float subClusDPhi(size_t clusNr)const;
  float subClusDEta(size_t clusNr)const;
  float subClusRawEnergyOverSCRawEnergy(SubClusNr clusNr)const{return subClusRawEnergyOverSCRawEnergy(static_cast<int>(clusNr));}
  float subClusRawEnergy(SubClusNr clusNr)const{return subClusRawEnergy(static_cast<int>(clusNr));}
  float subClusDPhi(SubClusNr clusNr)const{return subClusDPhi(static_cast<int>(clusNr));}
  float subClusDEta(SubClusNr clusNr)const{return subClusDEta(static_cast<int>(clusNr));}
  

  //modifiers
  void fill(const reco::SuperCluster& superClus,
	    const EcalRecHitCollection* ebRecHits,const EcalRecHitCollection* eeRecHits,
	    const CaloGeometry* geom,const CaloTopology* topology,
	    const reco::VertexCollection* vertices){
    fill(superClus,ebRecHits,eeRecHits,geom,topology,vertices->size());
  }
  void fill(const reco::SuperCluster& superClus,
	    const EcalRecHitCollection* ebRecHits,const EcalRecHitCollection* eeRecHits,
	    const CaloGeometry* geom,const CaloTopology* topology,
	    int nrVertices);
  void clear();
  
  //converts output to single vector for use in training
  void fillVec(std::vector<float>& inputVec)const;
  
private:
  //0 is obviously not a sensible energy for a supercluster so just return zero if this is the case
  float divideBySCRawEnergy_(float numer)const{return scRawEnergy()!=0 ? numer/scRawEnergy() : 0.;}
  void fillVecEB_(std::vector<float>& inputVec)const;
  void fillVecEE_(std::vector<float>& inputVec)const;
  
private:
  bool isEB_;
  
  //supercluster quantities
  float scRawEnergy_;
  float scCalibEnergy_;
  float scPreShowerEnergy_;
  float scEta_;
  float scPhi_;
  float scEtaWidth_;
  float scPhiWidth_;
  int scNrAdditionalClusters_; //excludes seed cluster

  //seed cluster quantities
  float seedClusEnergy_;
  float eMax_;
  float e2nd_;
  float e3x3_;
  float eTop_;
  float eBottom_;
  float eLeft_;
  float eRight_;
  float sigmaIEtaIEta_;
  float sigmaIEtaIPhi_;
  float sigmaIPhiIPhi_;

  //seed crystal quantities
  float seedCrysPhiOrY_;
  float seedCrysEtaOrX_;
  int seedCrysIEtaOrIX_;
  int seedCrysIPhiOrIY_;

  //sub cluster (non-seed) quantities
  float maxSubClusDR2_;
  float maxSubClusDRDPhi_;
  float maxSubClusDRDEta_;
  float maxSubClusDRRawEnergy_;
  std::vector<float> subClusRawEnergy_;
  std::vector<float> subClusDPhi_;
  std::vector<float> subClusDEta_;

  //event quantities
  int nrVtx_;
  
};

#endif
