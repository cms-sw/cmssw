#ifndef DATAFORMATS_METRECO_HALOCLUSTERCANDIDATEEB_H
#define DATAFORMATS_METRECO_HALOCLUSTERCANDIDATEEB_H



#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
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
class HaloClusterCandidateEB {

 public:
  HaloClusterCandidateEB();
  ~HaloClusterCandidateEB(){};

  

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
  //Specific to EB:
  edm::RefVector<EcalRecHitCollection>  getBeamHaloRecHitsCandidates() const {return bhrhcandidates;}
  double getEtStripIPhiSeedPlus1() const {return etstrip_iphiseedplus1;}
  double getEtStripIPhiSeedMinus1() const {return etstrip_iphiseedminus1;}
  double getHoverE() const {return hovere;}
  int getNbofCrystalsInEta() const {return numberofcrystalsineta;}

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
  //Specific to EB:
  void setBeamHaloRecHitsCandidates(edm::RefVector<EcalRecHitCollection>  x) {bhrhcandidates =x;}
  void setEtStripIPhiSeedPlus1(double x){ etstrip_iphiseedplus1=x;}
  void setEtStripIPhiSeedMinus1(double x){ etstrip_iphiseedminus1=x;}
  void setHoverE(double x){hovere=x;}
  void setNbofCrystalsInEta(double x){numberofcrystalsineta=x;}

 private:
  double et;
  double seed_et, seed_eta, seed_phi, seed_Z, seed_R, seed_time;
  double timediscriminator;
  bool ishalofrompattern;
  bool ishalofrompattern_hlt;
  //Specific to EB:
  double hovere;
  int numberofcrystalsineta;
  double etstrip_iphiseedplus1, etstrip_iphiseedminus1;
  edm::RefVector<EcalRecHitCollection>  bhrhcandidates;
    
};
 typedef std::vector<HaloClusterCandidateEB> HaloClusterCandidateEBCollection;
}
#endif
