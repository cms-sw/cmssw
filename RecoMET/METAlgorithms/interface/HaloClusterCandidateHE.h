#ifndef RECOMET_METALGORITHMS_HALOCLUSTERCANDIDATEHE_H
#define RECOMET_METALGORITHMS_HALOCLUSTERCANDIDATEHE_H



#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitFwd.h"
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



class HaloClusterCandidateHE {

 public:
  HaloClusterCandidateHE(){
    et =0;
    seed_et =0;
    seed_eta =0;
    seed_phi =0;
    seed_Z =0;
    seed_R =0;
    seed_time =0;
    timediscriminator =0;
    eoverh=0;
    h1overh123=0;
    etstrip_phiseedplus1=0;
    etstrip_phiseedminus1=0;
    clustersize=0;
  };
  ~HaloClusterCandidateHE(){};

  

  double GetClusterEt(){return et;}
  double GetSeedEt(){return seed_et;}
  double GetSeedEta(){return seed_eta;}
  double GetSeedPhi(){return seed_phi;}
  double GetSeedZ(){return seed_Z;}
  double GetSeedR(){return seed_R;}
  double GetSeedTime(){return seed_time;}
  double GetTimeDiscriminator(){return timediscriminator;}
  edm::RefVector<HBHERecHitCollection>  GetBeamHaloRecHitsCandidates(){return bhrhcandidates;}
  //Specific to HE:
  double SetEtStripIPhiSeedPlus1(){return etstrip_phiseedplus1;}
  double SetEtStripIPhiSeedMinus1(){return etstrip_phiseedminus1;}
  double GetEoverH(){return eoverh;}
  double GetH1overH123(){return h1overh123;}
  int GetClusterSize(){return clustersize;}
  
  
  void SetClusterEt(double x){ et=x;}
  void SetSeedEt(double x){ seed_et=x;}
  void SetSeedEta(double x){ seed_eta=x;}
  void SetSeedPhi(double x){ seed_phi=x;}
  void SetSeedZ(double x){ seed_Z=x;}
  void SetSeedR(double x){ seed_R=x;}
  void SetSeedTime(double x){ seed_time=x;}
  void SetTimeDiscriminator(double x){ timediscriminator=x;}
  void SetBeamHaloRecHitsCandidates(edm::RefVector<HBHERecHitCollection>  x) {bhrhcandidates =x;}
  void SetEtStripPhiSeedPlus1(double x){ etstrip_phiseedplus1=x;}
  void SetEtStripPhiSeedMinus1(double x){ etstrip_phiseedminus1=x;}
  void SetEoverH(double x){eoverh=x;}
  void SetH1overH123(double x){h1overh123=x;}
  void SetClusterSize(int x){clustersize=x;}

 private:
  double et;
  double seed_et, seed_eta, seed_phi, seed_Z, seed_R, seed_time;
  double timediscriminator;
  //Specific to HE:
  double eoverh,h1overh123;
  int clustersize;
  double etstrip_phiseedplus1, etstrip_phiseedminus1;
  edm::RefVector<HBHERecHitCollection>  bhrhcandidates;
  
};

#endif
