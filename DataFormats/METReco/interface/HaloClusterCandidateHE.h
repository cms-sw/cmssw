#ifndef DATAFORMATS_METRECO_HALOCLUSTERCANDIDATEHE_H
#define DATAFORMATS_METRECO_HALOCLUSTERCANDIDATEHE_H



#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/Event.h"


namespace reco {
class HaloClusterCandidateHE {

 public:
  HaloClusterCandidateHE();
  ~HaloClusterCandidateHE(){};

  

  double getClusterEt() const {return et;}
  double getSeedEt() const {return seed_et;}
  double getSeedEta() const {return seed_eta;}
  double getSeedPhi() const {return seed_phi;}
  double getSeedZ() const {return seed_Z;}
  double getSeedR() const {return seed_R;}
  double getSeedTime() const {return seed_time;}
  double getTimeDiscriminator() const {return timediscriminator;}
  bool getIsHaloFromPattern() const {return ishalofrompattern;}
  bool getIsHaloFromPattern_HLT() const {return ishalofrompattern_hlt;}
  //Specific to HE:
  edm::RefVector<HBHERecHitCollection>  getBeamHaloRecHitsCandidates() const {return bhrhcandidates;}
  double getEtStripPhiSeedPlus1() const {return etstrip_phiseedplus1;}
  double getEtStripPhiSeedMinus1() const {return etstrip_phiseedminus1;}
  double getEoverH() const {return eoverh;}
  double getH1overH123() const {return h1overh123;}
  int getClusterSize() const {return clustersize;}
  
  
  void setClusterEt(double x){ et=x;}
  void setSeedEt(double x){ seed_et=x;}
  void setSeedEta(double x){ seed_eta=x;}
  void setSeedPhi(double x){ seed_phi=x;}
  void setSeedZ(double x){ seed_Z=x;}
  void setSeedR(double x){ seed_R=x;}
  void setSeedTime(double x){ seed_time=x;}
  void setTimeDiscriminator(double x){ timediscriminator=x;}
  void setIsHaloFromPattern(bool x) { ishalofrompattern=x;}
  void setIsHaloFromPattern_HLT(bool x) { ishalofrompattern_hlt=x;}
  //Specific to HE:
  void setBeamHaloRecHitsCandidates(edm::RefVector<HBHERecHitCollection>  x) {bhrhcandidates =x;}
  void setEtStripPhiSeedPlus1(double x){ etstrip_phiseedplus1=x;}
  void setEtStripPhiSeedMinus1(double x){ etstrip_phiseedminus1=x;}
  void setEoverH(double x){eoverh=x;}
  void setH1overH123(double x){h1overh123=x;}
  void setClusterSize(int x){clustersize=x;}
  
 private:
  double et;
  double seed_et, seed_eta, seed_phi, seed_Z, seed_R, seed_time;
  double timediscriminator;
  bool ishalofrompattern;
  bool ishalofrompattern_hlt;
  //Specific to HE:
  double eoverh,h1overh123;
  int clustersize;
  double etstrip_phiseedplus1, etstrip_phiseedminus1;
  edm::RefVector<HBHERecHitCollection>  bhrhcandidates;
  
};
 typedef std::vector<HaloClusterCandidateHE> HaloClusterCandidateHECollection;
}
#endif
