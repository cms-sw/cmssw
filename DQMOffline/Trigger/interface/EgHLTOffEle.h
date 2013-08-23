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
#include "DataFormats/TrackReco/interface/Track.h"

#include "DQMOffline/Trigger/interface/EgHLTEgCutCodes.h"
#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"

namespace egHLT {
  class OffEle { 
    
  public:
    //helper struct to store the isolations
    struct IsolData {
      float em;
      float hadDepth1;
      float hadDepth2;
      float ptTrks;
      int nrTrks;
      //possibly going to move these to hlt data
      float hltHad;
      float hltTrksEle;
      float hltTrksPho;
      float hltEm;
    };
    
  public:
    //helper struct to store the cluster shapes
    struct ClusShapeData {
      float sigmaEtaEta;
      float sigmaIEtaIEta;  
      float sigmaPhiPhi;
      float sigmaIPhiIPhi; 
      float e1x5Over5x5;
      float e2x5MaxOver5x5;
      float r9;
    
    };
    
  public:
    //helper struct to store reco approximations of variables made by HLT - and HLT p4 to get eta,phi
    struct HLTData {
      float dEtaIn;
      float dPhiIn;  
      float invEInvP;
      //math::XYZTLorentzVector p4;
      float HLTeta;
      float HLTphi;
      float HLTenergy;
    };
    
  public:
    //helper struct to store event-wide variables
    struct EventData {
      int NVertex;
    };

  private:
    const reco::GsfElectron* gsfEle_; //pointers to the underlying electron (we do not own this)

    ClusShapeData clusShapeData_;
    IsolData isolData_;
    HLTData hltData_;
    EventData eventData_;

    //these are bit-packed words telling me which cuts the electron fail (ie 0x0 is passed all cuts)
    int cutCode_;
    int looseCutCode_;
    //the idea is that these are user definable cuts meant to be idenital to the specified trigger
    //it is probably clear to the reader that I havent decided on the most efficient way to do this
    std::vector<std::pair<TrigCodes::TrigBitSet,int> > trigCutsCutCodes_; //unsorted vector (may sort if have performance issues)
  
    //and these are the trigger bits stored
    //note that the trigger bits are defined at the begining of each job
    //and do not necessaryly map between jobs
    TrigCodes::TrigBitSet trigBits_;
    
  public:
    
    OffEle(const reco::GsfElectron& ele,const ClusShapeData& shapeData,const IsolData& isolData,const HLTData& hltData,const EventData& eventData):
      gsfEle_(&ele),clusShapeData_(shapeData),isolData_(isolData),hltData_(hltData),eventData_(eventData),
      cutCode_(int(EgCutCodes::INVALID)),looseCutCode_(int(EgCutCodes::INVALID)){}
    ~OffEle(){}
    

    //modifiers  
    int NVertex()const{return eventData_.NVertex;}
    void setCutCode(int code){cutCode_=code;}
    void setLooseCutCode(int code){looseCutCode_=code;} 
    //slightly inefficient way, think I can afford it and its a lot easier to just make the sorted vector outside the class
    void setTrigCutsCutCodes(const std::vector<std::pair<TrigCodes::TrigBitSet,int> >& trigCutsCutCodes){trigCutsCutCodes_=trigCutsCutCodes;}
    void setTrigBits(TrigCodes::TrigBitSet bits){trigBits_=bits;}
    
    const reco::GsfElectron* gsfEle()const{return gsfEle_;}

    //kinematic and geometric methods
    float et()const{return gsfEle_->et();} 
    // float et()const{return etSC();}
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
    bool isGap()const{return gsfEle_->isEBGap() || gsfEle_->isEEGap() || gsfEle_->isEBEEGap();}
    
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
    float dEtaOut()const{return gsfEle_->deltaEtaSeedClusterTrackAtCalo();}
    float epIn()const{return gsfEle_->eSuperClusterOverP();}
    float epOut()const{return gsfEle_->eSeedClusterOverPout();}
    
    //variables with no direct method
    float sigmaEtaEta()const;
    float sigmaEtaEtaUnCorr()const{return clusShapeData_.sigmaEtaEta;}
    float sigmaIEtaIEta()const{return clusShapeData_.sigmaIEtaIEta;}					
    float sigmaPhiPhi()const{return clusShapeData_.sigmaPhiPhi;}
    //float sigmaIPhiIPhi()const{return clusShapeData_.sigmaIPhiIPhi;}
    float e2x5MaxOver5x5()const{return clusShapeData_.e2x5MaxOver5x5;}
    float e1x5Over5x5()const{return clusShapeData_.e1x5Over5x5;}
									
    float r9()const{return clusShapeData_.r9;}
    //float sigmaPhiPhi()const{return clusShape_!=NULL ? sqrt(clusShape_->covPhiPhi()) : 999;}
    float bremFrac()const{return (pVtx()-pCalo())/pVtx();}
    float invEInvP()const{return gsfEle_->caloEnergy()!=0 && gsfEle_->trackMomentumAtVtx().R()!=0. ? 1./gsfEle_->caloEnergy() - 1./gsfEle_->trackMomentumAtVtx().R() : -999;}
    //float e9OverE25()const{return clusShape_!=NULL ? clusShape_->e3x3()/clusShape_->e5x5() : -999;}
    
    //isolation
    float isolEm()const{return isolData_.em;}
    float isolHad()const{return isolHadDepth1()+isolHadDepth2();}
    float isolHadDepth1()const{return isolData_.hadDepth1;}
    float isolHadDepth2()const{return isolData_.hadDepth2;}
    float isolPtTrks()const{return isolData_.ptTrks;}
    int isolNrTrks()const{return isolData_.nrTrks;}
    float hltIsolTrksEle()const{return isolData_.hltTrksEle;}
    float hltIsolTrksPho()const{return isolData_.hltTrksPho;}
    float hltIsolHad()const{return isolData_.hltHad;}
    float hltIsolEm()const{return isolData_.hltEm;}
    
    //some hlt id variables (note these are reco approximations)
    float hltDEtaIn()const{return hltData_.dEtaIn;}
    float hltDPhiIn()const{return hltData_.dPhiIn;}
    float hltInvEInvP()const{return hltData_.invEInvP;}
    //hlt position - not a reco approximation, taken from triggerobject
    //const math::XYZTLorentzVector& HLTp4()const{return hltData_.p4;}
    float hltPhi()const{return hltData_.HLTphi;}
    float hltEta()const{return hltData_.HLTeta;}
    float hltEnergy()const{return hltData_.HLTenergy;}
    //Diference between HLT Et and reco SC Et
    float DeltaE()const{return (hltEnergy() - caloEnergy());}

    //ctf track accessor and validatity checker
    reco::TrackRef ctfTrack()const{return gsfEle_->closestCtfTrackRef();} //in theory lightweight (if they follow good design),return by value
    //track is only valid if it exists and track extra exists (track extra is only stored in reco)
    bool validCTFTrack()const{return gsfEle_->closestCtfTrackRef().isNonnull() && gsfEle_->closestCtfTrackRef()->extra().isNonnull();}
    

    //ctf track varibles, used as hlt uses this algo
    float ctfTrkP()const{return validCTFTrack() ? ctfTrack()->p() : -999.;}
    float ctfTrkPt()const{return validCTFTrack() ? ctfTrack()->pt() : -999.;}
    float ctfTrkEta()const{return validCTFTrack() ? ctfTrack()->eta() : -999.;}
    float ctfTrkChi2()const{return validCTFTrack() ? ctfTrack()->chi2() : 999.;}
    float ctfTrkNDof()const{return validCTFTrack() ? ctfTrack()->ndof() : 999.;} //this will give chi2/ndof a valid value, perhaps rethink
    float ctfTrkPtOuter()const{return validCTFTrack() ?  ctfTrack()->outerMomentum().Perp2() : -999.;}
    float ctfTrkPtInner()const{return validCTFTrack() ?  ctfTrack()->innerMomentum().Perp2() : -999.;}
    float ctfTrkInnerRadius()const{return validCTFTrack() ? ctfTrack()->innerPosition().Rho() : 999.;}
    float ctfTrkOuterRadius()const{return validCTFTrack() ? ctfTrack()->outerPosition().Rho() : -999.;}
    int ctfTrkHitsFound()const{return validCTFTrack() ? static_cast<int>(ctfTrack()->found()) : -999;}
    int ctfTrkHitsLost()const{return validCTFTrack() ? static_cast<int>(ctfTrack()->lost()) : -999;}
    int ctfTrkNrHits()const{return validCTFTrack() ? static_cast<int>(ctfTrack()->recHitsSize()) : -999;}

    //selection cuts
    int cutCode()const{return cutCode_;}
    int looseCutCode()const{return looseCutCode_;}
   
    
    //trigger codes are just used as a unique identifier of the trigger, it is an error to specify more than a single bit
    //the idea here is to allow an arbitary number of electron triggers
    int trigCutsCutCode(const TrigCodes::TrigBitSet& trigger)const; 
    //trigger
    TrigCodes::TrigBitSet trigBits()const{return trigBits_;}
    
    
  };

}



#endif
