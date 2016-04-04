#ifndef DATAFORMATS_METRECO_HALOCLUSTERCANDIDATEEE_H
#define DATAFORMATS_METRECO_HALOCLUSTERCANDIDATEEE_H



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
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/Event.h"


namespace reco {
class HaloClusterCandidateEE {

 public:
  HaloClusterCandidateEE();
  ~HaloClusterCandidateEE(){};

  

  double getClusterEt() const {return et;}
  double getSeedEt() const {return seed_et;}
  double getSeedEta() const {return seed_eta;}
  double getSeedPhi() const {return seed_phi;}
  double getSeedZ() const {return seed_Z;}
  double getSeedR() const {return seed_R;}
  double getSeedTime() const {return seed_time;}
  double getTimeDiscriminator() const {return timediscriminator;}
  double getIsHaloFromPattern() const {return ishalofrompattern;}
  double getIsHaloFromPattern_HLT() const {return ishalofrompattern_hlt;}
  //Specific to EE:
  edm::RefVector<EcalRecHitCollection>  getBeamHaloRecHitsCandidates() const {return bhrhcandidates;}
  double getH2overE() const {return h2overe;}
  int getNbEarlyCrystals() const {return nbearlycrystals;}
  int getNbLateCrystals() const {return nblatecrystals;}
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
  //Specific to EE:
  void setBeamHaloRecHitsCandidates(edm::RefVector<EcalRecHitCollection>  x) {bhrhcandidates =x;}
  void setH2overE(double x){h2overe=x;}
  void setNbEarlyCrystals(int x){nbearlycrystals=x;}
  void setNbLateCrystals(int x){nblatecrystals=x;}
  void setClusterSize(int x){clustersize=x;}

 private:
  double et;
  double seed_et, seed_eta, seed_phi, seed_Z, seed_R, seed_time;
  double timediscriminator;
  bool ishalofrompattern;
  bool ishalofrompattern_hlt;
  //Specific to EE:
  double h2overe;
  int nbearlycrystals, nblatecrystals, clustersize;
  edm::RefVector<EcalRecHitCollection>  bhrhcandidates;

};

 typedef std::vector<HaloClusterCandidateEE> HaloClusterCandidateEECollection;

}
#endif
