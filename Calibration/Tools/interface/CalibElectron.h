#ifndef CALIBELECTRON_H
#define CALIBELECTRON_H

#include <TROOT.h>
#include <TLorentzVector.h>

#include <vector>
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"

namespace calib
{
  class CalibElectron {
    
  public:
    
    CalibElectron();
    CalibElectron(const reco::PixelMatchGsfElectron* ele ,const EcalRecHitCollection* theHits) : 
      theElectron_(ele),
      theHits_(theHits) 
      {
      };

    ~CalibElectron() {};


    std::vector< std::pair<int,float> > getCalibModulesWeights(TString calibtype);
    const reco::PixelMatchGsfElectron* getRecoElectron() { return theElectron_; }
    const EcalRecHitCollection* getRecHits() { return theHits_; }

/*     int getElectronClass() { return electronClass_;}; */

/*     bool isInCrack(); */
/*     bool isInBarrel(); */
/*     bool isInEndcap(); */

/*     float fNCrystals(int nCry); */
/*     float fEtaBarrelBad(float scEta); */
/*     float fEtaBarrelGood(float scEta); */
/*     float fEtaBarrelGood25(float scEta); */
/*     float fEtaEndcap(float scEta); */
/*     int nCrystalsGT2Sigma(const reco::SuperCluster* seed,float sigmaNoise); */

    /*   inline TVector*  GetWeights(){ return &weights_; };  */
    /*   inline IntVec*  GetModules() { return &modules_; }; */
    /*   inline pair<Int_t, Float_t> GetModuleWeight(UInt_t); */
    /*   inline UInt_t GetNoModules(){ return nmodules_; }; */
    /*   inline Float_t GetEnergy() { return pvec_.Energy(); }; */
    /*    TLorentzVector getEnergy4Vector();  */
    /*   inline TLorentzVector* GetTrueEnergy4Vector() { return &true_pvec_; }; */
    /*   inline Float_t GetS25() { return S25_;}; */
    /*   inline Float_t GetS9() { return S9_;}; */

    /*   inline void SetEnergy(Float_t energy){pvec_.SetE(energy);}; */
    /*   inline void SetEnergy4Vector(TLorentzVector vec){pvec_=vec;}; */
    /*   inline void SetTrueEnergy4Vector(TLorentzVector vec){true_pvec_=vec;}; */
    /*   inline void SetEnergy4Vector(Float_t energy,Float_t eta,Float_t phi); */
    /*   inline void SetWeightsModules(const TVector& weights,const IntVec& modules); */

  private:
  
    //  VClusteringAlgo* theAlgo;
    const reco::PixelMatchGsfElectron* theElectron_;
    
    const EcalRecHitCollection* theHits_;

/*     DetId maxEnergyHitId_; */

/*     Float_t getE25(); */
/*     Float_t getE9(); */
/*     Float_t getE1(); */

/*     void fillTrackInfo(const reco::PixelMatchGsfElectron* anEle); */
/*     void fillSCInfo(const reco::Electron* anEle,const EcalRecHitCollection* myHits); */
    //  bool isClusterized;
    //  ClassDef(Electron,1);  
  };
}
#endif

