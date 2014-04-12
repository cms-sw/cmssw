#ifndef DQMOFFLINE_TRIGGER_EGHLTOFFPHO
#define DQMOFFLINE_TRIGGER_EGHLTOFFPHO

//class: EgHLTOffPho
//
//author: Sam Harper (July 2008)
//
//
//aim: to allow easy access to phoctron ID variables
//     currently the CMSSW photon classes are a mess with key photon variables not being accessable from Photon
//     this a stop gap to produce a simple photon class with all variables easily accessable via methods 
//     note as this is meant for HLT Offline DQM, I do not want the overhead of converting to pat
//
//implimentation: aims to be a wrapper for Photon methods, it is hoped that in time these methods will be directly added to Photon and so
//                make this class obsolute
//                unfortunately can not be a pure wrapper as needs to store isol and cluster shape
//


#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h" 

#include "DQMOffline/Trigger/interface/EgHLTEgCutCodes.h"
#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"

namespace egHLT {
  class OffPho { 
    
  public:
    //helper struct to store the isolations
    struct IsolData {
      int nrTrks;
      float ptTrks;
      float em;
      float had;
      float hltHad;
      float hltTrks;
      float hltEm;
    };
    
  public:
    //helper struct to store the cluster shapes
    struct ClusShapeData {
      float sigmaEtaEta;
      float sigmaIEtaIEta;
      float e2x5MaxOver5x5; 
      float e1x5Over5x5;
      float sigmaPhiPhi;
      float sigmaIPhiIPhi;
      float r9;
    };
    
public:
    //helper struct to store reco approximations of variables made by HLT
    struct HLTData {
      //const math::XYZTLorentzVector p4() const;
      float HLTeta;
      float HLTphi;
      float HLTenergy;
    };    


  private:
    const reco::Photon* pho_; //pointers to the underlying phoctron (we do not own this)

    ClusShapeData clusShapeData_;
    IsolData isolData_;
    HLTData hltData_;
    
    //these are bit-packed words telling me which cuts the photon fail (ie 0x0 is passed all cuts) 
    int cutCode_;
    int looseCutCode_;
  
    //the idea is that these are user definable cuts mean to be idenital to the specified trigger
    //it is probably clear to the reader that I havent decided on the most efficient way to do this
    std::vector<std::pair<TrigCodes::TrigBitSet,int> > trigCutsCutCodes_; //unsorted vector (may sort if have performance issues)
    
    //and these are the trigger bits stored
    //note that the trigger bits are defined at the begining of each job
    //and do not necessaryly map between jobs
    TrigCodes::TrigBitSet trigBits_;
    
  public:
    
    OffPho(const reco::Photon& pho,const ClusShapeData& shapeData,const IsolData& isolData,const HLTData& hltData):
      pho_(&pho),clusShapeData_(shapeData),isolData_(isolData),hltData_(hltData),
      cutCode_(int(EgCutCodes::INVALID)),looseCutCode_(int(EgCutCodes::INVALID)){}
    ~OffPho(){}
    
    //modifiers  
    void setCutCode(int code){cutCode_=code;}
    void setLooseCutCode(int code){looseCutCode_=code;} 
    
    //slightly inefficient way, think I can afford it and its a lot easier to just make the sorted vector outside the class
    void setTrigCutsCutCodes(const std::vector<std::pair<TrigCodes::TrigBitSet,int> >& trigCutsCutCodes){trigCutsCutCodes_=trigCutsCutCodes;}
    void setTrigBits(TrigCodes::TrigBitSet bits){trigBits_=bits;}
    
    const reco::Photon* recoPho()const{return pho_;}

    //kinematic and geometric methods
    float et()const{return pho_->et();}  
    float pt()const{return pho_->pt();}
    float energy()const{return pho_->energy();}
    float eta()const{return pho_->eta();}
    float phi()const{return pho_->phi();}
    float etSC()const{return pho_->superCluster()->position().rho()/pho_->superCluster()->position().r()*energy();}
    float etaSC()const{return pho_->superCluster()->eta();}
    float detEta()const{return etaSC();}
    float phiSC()const{return pho_->superCluster()->phi();}
    float zVtx()const{return pho_->vz();}
    const math::XYZTLorentzVector& p4()const{return pho_->p4();}
    
    bool isGap()const{return pho_->isEBGap() || pho_->isEEGap() || pho_->isEBEEGap();}
    
    //abreviations of overly long Photon methods, I'm sorry but if you cant figure out what hOverE() means, you shouldnt be using this class
    float hOverE()const{return pho_->hadronicOverEm();}
    
    
    
    float sigmaEtaEta()const;
    float sigmaEtaEtaUnCorr()const{return clusShapeData_.sigmaEtaEta;}
    float sigmaIEtaIEta()const{return clusShapeData_.sigmaIEtaIEta;}					
    float sigmaPhiPhi()const{return clusShapeData_.sigmaPhiPhi;}
    float sigmaIPhiIPhi()const{return clusShapeData_.sigmaIPhiIPhi;}
    float e2x5MaxOver5x5()const{return clusShapeData_.e2x5MaxOver5x5;}
    float e1x5Over5x5()const{return clusShapeData_.e1x5Over5x5;}
    float r9()const{return clusShapeData_.r9;}
    
    //isolation
    float isolEm()const{return isolData_.em;}
    float isolHad()const{return isolData_.had;}
    int isolNrTrks()const{return isolData_.nrTrks;}
    float isolPtTrks()const{return isolData_.ptTrks;}
    float hltIsolHad()const{return isolData_.hltHad;}
    float hltIsolTrks()const{return isolData_.hltTrks;}
    float hltIsolEm()const{return isolData_.hltEm;}

    //hlt position - not a reco approximation, taken from triggerobject
    //const math::XYZTLorentzVector& HLTp4()const{return hltDataPho_.p4();}
    float hltPhi()const{return hltData_.HLTphi;}
    float hltEta()const{return hltData_.HLTeta;}
    float hltEnergy()const{return hltData_.HLTenergy;}
    //Diference between HLT Et and reco SC Et
    float DeltaE()const{return (hltEnergy() - energy());}

    //selection cuts
    int cutCode()const{return cutCode_;}
    int looseCutCode()const{return looseCutCode_;}
  
    //trigger codes are just used as a unique identifier of the trigger, it is an error to specify more than a single bit
    //the idea here is to allow an arbitary number of photon triggers
    int trigCutsCutCode(const TrigCodes::TrigBitSet& trigger)const; 
    
    //trigger
    TrigCodes::TrigBitSet trigBits()const{return trigBits_;}
    
  };
}

#endif
