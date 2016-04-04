#ifndef RECOMET_METALGORITHMS_HALOCLUSTERCANDIDATEEB_H
#define RECOMET_METALGORITHMS_HALOCLUSTERCANDIDATEEB_H



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



class HaloClusterCandidateEB {

 public:
  HaloClusterCandidateEB(){
    et =0;
    seed_et =0;
    seed_eta =0;
    seed_phi =0;
    seed_Z =0;
    seed_R =0;
    seed_time =0;
    timediscriminator =0;
    hovere=0;
    numberofcrystalsineta=0;
    etstrip_iphiseedplus1=0;
    etstrip_iphiseedminus1=0;

  };
  ~HaloClusterCandidateEB(){};

  

  double GetClusterEt(){return et;}
  double GetSeedEt(){return seed_et;}
  double GetSeedEta(){return seed_eta;}
  double GetSeedPhi(){return seed_phi;}
  double GetSeedZ(){return seed_Z;}
  double GetSeedR(){return seed_R;}
  double GetSeedTime(){return seed_time;}
  double GetTimeDiscriminator(){return timediscriminator;}
  edm::RefVector<EcalRecHitCollection>  GetBeamHaloRecHitsCandidates(){return bhrhcandidates;}
  //Specific to EB:
  double GetEtStripIPhiSeedPlus1(){return etstrip_iphiseedplus1;}
  double GetEtStripIPhiSeedMinus1(){return etstrip_iphiseedminus1;}
  double GetHoverE(){return hovere;}
  int GetNbofCrystalsInEta(){return numberofcrystalsineta;}

  
  void SetClusterEt(double x){ et=x;}
  void SetSeedEt(double x){ seed_et=x;}
  void SetSeedEta(double x){ seed_eta=x;}
  void SetSeedPhi(double x){ seed_phi=x;}
  void SetSeedZ(double x){ seed_Z=x;}
  void SetSeedR(double x){ seed_R=x;}
  void SetSeedTime(double x){ seed_time=x;}
  void SetTimeDiscriminator(double x){ timediscriminator=x;}
  void SetBeamHaloRecHitsCandidates(edm::RefVector<EcalRecHitCollection>  x) {bhrhcandidates =x;}

  //Specific to EB:
  void SetEtStripIPhiSeedPlus1(double x){ etstrip_iphiseedplus1=x;}
  void SetEtStripIPhiSeedMinus1(double x){ etstrip_iphiseedminus1=x;}
  void SetHoverE(double x){hovere=x;}
  void SetNbofCrystalsInEta(double x){numberofcrystalsineta=x;}


 private:
  double et;
  double seed_et, seed_eta, seed_phi, seed_Z, seed_R, seed_time;
  double timediscriminator;
  //Specific to EB:
  double hovere;
  int numberofcrystalsineta;
  double etstrip_iphiseedplus1, etstrip_iphiseedminus1;
  
  edm::RefVector<EcalRecHitCollection>  bhrhcandidates;
  

  
};

#endif
