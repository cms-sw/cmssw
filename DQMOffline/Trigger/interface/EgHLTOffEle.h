#ifndef DQMOFFLINE_TRIGGER_EGHLTOFFELE
#define DQMOFFLINE_TRIGGER_EGHLTOFFELE

//class: EgHLTOffEle
//
//author: Sam Harper (July 2008)
//
//
//aim: to allow easy access to electron ID variables
//     currently the CMSSW electron classes are a mess with key electron selection variables not being accessable from GsfElectron
//     this a stop gap to produce a simple electron class with all variables easily accessable via methods 
//     note as this is meant for HLT Offline DQM, I do not want the overhead of converting to pat
//
//implimentation: aims to be a wrapper for GsfElectron methods, it is hoped that in time these methods will be directly added to GsfElectron and so
//                make this class obsolute
//                unfortunately can not be a pure wrapper as needs to store isol and cluster shape
//


#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"

#include "DQMOffline/Trigger/interface/CutCodes.h"
#include "DQMOffline/Trigger/interface/TrigCodes.h"

class EgHLTOffEle { 
  
 public:
  //helper struct to store the isolations
  struct IsolData {
    float nrTrks;
    float ptTrks;
    float em;
    float had;
  };

public:
  //helper struct to store the cluster shapes
  struct ClusShapeData {
    float sigmaEtaEta;
    //float sigmaIEtaIEta;
    //float e2x5MaxOver5x5;
    float sigmaPhiPhi;
    //float sigmaIPhiIPhi;
  };


 private:
  const reco::GsfElectron* gsfEle_; //pointers to the underlying electron (we do not own this)
  // const reco::ClusterShape* clusShape_; //pointers to the underlying cluster shape (we do not own this and it may be null)
  ClusShapeData clusShapeData_;
  IsolData isolData_;

  //these are bit-packed words telling me which cuts the electron fail (ie 0x0 is passed all cuts)
  int tagCutCode_;
  int probeCutCode_;
  int cutCode_;
  
  //and these are the trigger bits stored
  //note that the trigger bits are defined at the begining of each job
  //and do not necessaryly map between jobs
  TrigCodes::TrigBitSet trigBits_;

 public:
  
 EgHLTOffEle(const reco::GsfElectron& ele,const ClusShapeData& shapeData,const IsolData& isolData):
   gsfEle_(&ele),clusShapeData_(shapeData),isolData_(isolData),
   tagCutCode_(CutCodes::INVALID),probeCutCode_(CutCodes::INVALID),cutCode_(CutCodes::INVALID){}
  ~EgHLTOffEle(){}
  
  //modifiers  
  void setTagCutCode(int code){tagCutCode_=code;}
  void setProbeCutCode(int code){probeCutCode_=code;} 
  void setCutCode(int code){cutCode_=code;}
  void setTrigBits(TrigCodes::TrigBitSet bits){trigBits_=bits;}

  //kinematic and geometric methods
  float et()const{return gsfEle_->et();}
  float energy()const{return gsfEle_->energy();}
  float eta()const{return gsfEle_->eta();}
  float phi()const{return gsfEle_->phi();}
  float etSC()const{return gsfEle_->superCluster()->position().rho()/gsfEle_->superCluster()->position().r()*caloEnergy();}
  float caloEnergy()const{return gsfEle_->caloEnergy();}
  float etaSC()const{return gsfEle_->superCluster()->eta();}
  float detEta()const{return etaSC();}
  float phiSC()const{return gsfEle_->superCluster()->phi();}
  float zVtx()const{return gsfEle_->TrackPositionAtVtx().z();}
  const math::XYZTLorentzVector& p4()const{return gsfEle_->p4();}

  //classification (couldnt they have just named it 'type')
  int classification()const{return gsfEle_->classification();}

  //track methods
  int charge()const{return gsfEle_->charge();}
  float pVtx()const{return gsfEle_->trackMomentumAtVtx().R();}
  float pCalo()const{return gsfEle_->trackMomentumAtCalo().R();}
  float ptVtx()const{return gsfEle_->trackMomentumAtVtx().rho();}
  float ptCalo()const{return gsfEle_->trackMomentumAtCalo().rho();}
 
  
  //abreviations of overly long GsfElectron methods, I'm sorry but if you cant figure out what hOverE() means, you shouldnt be using this class
  float hOverE()const{return gsfEle_->hadronicOverEm();}
  float dEtaIn()const{return gsfEle_->deltaEtaSuperClusterTrackAtVtx();}
  float dPhiIn()const{return gsfEle_->deltaPhiSuperClusterTrackAtVtx();}
  float dPhiOut()const{return gsfEle_->deltaPhiSeedClusterTrackAtCalo();}
  float epIn()const{return gsfEle_->eSuperClusterOverP();}
  float epOut()const{return gsfEle_->eSeedClusterOverPout();}
  
  //variables with no direct method
  float sigmaEtaEta()const;
  float sigmaEtaEtaUnCorr()const{return clusShapeData_.sigmaEtaEta;}
  //float sigmaIEtaIEta()const;						
  float sigmaPhiPhi()const{return clusShapeData_.sigmaPhiPhi;}
  //float sigmaIPhiIPhi()const{return clusShapeData_.sigmaIPhiIPhi;}
  //float e2x5MaxOver5x5()const{return clusShapeData_.e2x5MaxOver5x5;}
  //float sigmaPhiPhi()const{return clusShape_!=NULL ? sqrt(clusShape_->covPhiPhi()) : 999;}
  float bremFrac()const{return (pVtx()-pCalo())/pVtx();}
  float invEOverInvP()const{return 1./gsfEle_->caloEnergy() - 1./gsfEle_->trackMomentumAtVtx().R();}
  //float e9OverE25()const{return clusShape_!=NULL ? clusShape_->e3x3()/clusShape_->e5x5() : -999;}
  
  //isolation
  float isolEm()const{return isolData_.em;}
  float isolHad()const{return isolData_.had;}
  float isolNrTrks()const{return isolData_.nrTrks;}
  float isolPtTrks()const{return isolData_.ptTrks;}

  //selection cuts
  int tagCutCode()const{return tagCutCode_;}
  int probeCutCode()const{return probeCutCode_;}
  int cutCode()const{return cutCode_;}

  //trigger
  TrigCodes::TrigBitSet trigBits()const{return trigBits_;}
  

};

#endif
