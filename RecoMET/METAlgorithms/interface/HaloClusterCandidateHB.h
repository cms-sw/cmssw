#ifndef RECOMET_METALGORITHMS_HALOCLUSTERCANDIDATEHB_H
#define RECOMET_METALGORITHMS_HALOCLUSTERCANDIDATEHB_H



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



class HaloClusterCandidateHB {

 public:
  HaloClusterCandidateHB(){
    et =0;
    seed_et =0;
    seed_eta =0;
    seed_phi =0;
    seed_Z =0;
    seed_R =0;
    seed_time =0;
    timediscriminatoritbh =0;
    timediscriminatorotbh =0;
    eoverh=0;
    nbtowersineta=0;
    etstrip_phiseedplus1=0;
    etstrip_phiseedminus1=0;
  };
  ~HaloClusterCandidateHB(){};

  

  double GetClusterEt(){return et;}
  double GetSeedEt(){return seed_et;}
  double GetSeedEta(){return seed_eta;}
  double GetSeedPhi(){return seed_phi;}
  double GetSeedZ(){return seed_Z;}
  double GetSeedR(){return seed_R;}
  double GetSeedTime(){return seed_time;}
  edm::RefVector<HBHERecHitCollection>  GetBeamHaloRecHitsCandidates(){return bhrhcandidates;}
  //Specific to HB:
  double GetEoverH(){return eoverh;}
  int GetNbTowersInEta(){return nbtowersineta;}
  double GetTimeDiscriminatorITBH(){return timediscriminatoritbh;}
  double GetTimeDiscriminatorOTBH(){return timediscriminatorotbh;}
  double GetEtStripPhiSeedPlus1(){return etstrip_phiseedplus1;}
  double GetEtStripPhiSeedMinus1(){return etstrip_phiseedminus1;}

  
  
  void SetClusterEt(double x){ et=x;}
  void SetSeedEt(double x){ seed_et=x;}
  void SetSeedEta(double x){ seed_eta=x;}
  void SetSeedPhi(double x){ seed_phi=x;}
  void SetSeedZ(double x){ seed_Z=x;}
  void SetSeedR(double x){ seed_R=x;}
  void SetSeedTime(double x){ seed_time=x;}
  void SetBeamHaloRecHitsCandidates(edm::RefVector<HBHERecHitCollection>  x) {bhrhcandidates =x;}
  //Specific to HB: 
  void SetEoverH(double x){eoverh=x;}
  void SetNbTowersInEta(double x){nbtowersineta=x;}
  void SetTimeDiscriminatorITBH(double x){ timediscriminatoritbh=x;}
  void SetTimeDiscriminatorOTBH(double x){ timediscriminatorotbh=x;}
  void SetEtStripPhiSeedPlus1(double x){etstrip_phiseedplus1=x;}
  void SetEtStripPhiSeedMinus1(double x){etstrip_phiseedminus1=x;}

 private:
  double et;
  double seed_et, seed_eta, seed_phi, seed_Z, seed_R, seed_time;


  //Specific to HB:
  double timediscriminatoritbh, timediscriminatorotbh;
  double eoverh;
  int nbtowersineta;
  double etstrip_phiseedplus1, etstrip_phiseedminus1;
  edm::RefVector<HBHERecHitCollection>  bhrhcandidates;
  
};

#endif
