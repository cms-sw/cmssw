#ifndef DQMOFFLINE_TRIGGER_EGHLTDQMCUT
#define DQMOFFLINE_TRIGGER_EGHLTDQMCUT 

//class: EgHLTDQMCut
//
//author: Sam Harper (June 2008)
//
//WARNING: interface is NOT final, please dont use this class for now without clearing it with me
//         as I will change it and possibly break all your code
//
//aim: to allow the user to place a cut on the electron using it or the event
//
//implimentation:

#include "DQMOffline/Trigger/interface/EgHLTOffEvt.h"
#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"

#include "DataFormats/Math/interface/deltaR.h"

//this is a pure virtual struct which defines the interface to the cut objects
//it is also currently uncopyable

namespace egHLT {

  template<class T> struct EgHLTDQMCut { 
    
    private: 
    //disabling copy and assignment for all objects
    EgHLTDQMCut& operator=(const EgHLTDQMCut& rhs){return *this;}
      protected:
      //only derived classes can call the copy constructor (needed for clone...)
      EgHLTDQMCut(const EgHLTDQMCut& rhs){}
    
    public:
    EgHLTDQMCut(){}
    virtual ~EgHLTDQMCut(){}
    virtual bool pass(const T& obj,const OffEvt& evt)const=0;
    virtual EgHLTDQMCut<T>* clone()const=0; //caller owns the pointer
  };
  
  
  
  
  template<class T> struct EgHLTDQMVarCut : public EgHLTDQMCut<T> {
    private:
    int cutsToPass_; //the cuts whose eff we are measuring
    int (T::*cutCodeFunc_)()const;
    
    public:
    EgHLTDQMVarCut(int cutsToPass,int (T::*cutCodeFunc)()const):cutsToPass_(cutsToPass),cutCodeFunc_(cutCodeFunc){}
    ~EgHLTDQMVarCut(){}
    
    bool pass(const T& obj,const OffEvt& evt)const;
    EgHLTDQMCut<T>* clone()const{return new EgHLTDQMVarCut(*this);} //default copy constructor is fine
    
  };
  
  //to understand this you need to know about
  //1) templates
  //2) inheritance (sort of)
  //3) function pointers
  //4) bitwise operations
  //All it does is get the bitword corresponding to the cuts the electron failed and the mask the bits which correspond to cuts we dont care about and then see if any bits are still set
  template<class T> bool EgHLTDQMVarCut<T>::pass(const T& obj,const OffEvt& evt)const
    {
      if(((obj.*cutCodeFunc_)() & cutsToPass_)==0) return true;
      else return false;
    }
  
  
  //now this is similar to EgHLTDQMVarCut except that it allows a key to specified to the cut code function
  template<class T,class Key> struct EgHLTDQMUserVarCut : public EgHLTDQMCut<T> {
    private:
    
    int (T::*cutCodeFunc_)(const Key&)const;   
    const Key key_;
    int cutsNotToMask_;
    
    public:
    EgHLTDQMUserVarCut(int (T::*cutCodeFunc)(const Key&)const,const Key& key,int cutsNotToMask=~0x0):cutCodeFunc_(cutCodeFunc),key_(key),cutsNotToMask_(cutsNotToMask){}
    ~EgHLTDQMUserVarCut(){}
    
    bool pass(const T& obj,const OffEvt& evt)const;
    EgHLTDQMCut<T>* clone()const{return new EgHLTDQMUserVarCut(*this);} //default copy constructor is fine
    
  };
  
  template<class T,class Key> bool EgHLTDQMUserVarCut<T,Key>::pass(const T& obj,const OffEvt& evt)const
    { 
      if(((obj.*cutCodeFunc_)(key_) & cutsNotToMask_)==0) return true;
      else return false;
    }
  
  template<class T,typename varType> struct EgGreaterCut : public EgHLTDQMCut<T> {
    private:
    varType cutValue_;
    varType (T::*varFunc_)()const;
 
    public:
    
    EgGreaterCut(varType cutValue,varType (T::*varFunc)()const):
      cutValue_(cutValue),varFunc_(varFunc){}
    
    bool pass(const T& obj,const OffEvt& evt)const{return (obj.*varFunc_)()>cutValue_;}
    EgHLTDQMCut<T>* clone()const{return new EgGreaterCut(*this);} //default copy constructor is fine
  };
  
  //this struct allows multiple cuts to be strung together
  //now I could quite simply have a link to the next cut defined in the base class 
  //and the operator<< defined there also but I dont for the following reason
  //1) I'm concerned about a circular chain of cuts (hence why you cant do a EgMultiCut << EgMultiCut)
  //2) it requires all cuts to multi cut aware in the pass function
  //in the future I may change it so this class isnt required
  template<class T> struct EgMultiCut : public EgHLTDQMCut<T> {
    private:
    std::vector<const EgHLTDQMCut<T>*> cuts_;//all the points to the cuts we own 
    
    public:
    EgMultiCut(){}  
    EgMultiCut(const EgMultiCut<T>& rhs);
    ~EgMultiCut(){for(size_t i=0;i<cuts_.size();i++) delete cuts_[i];}
    
    //we own any cut given to use this way
    EgMultiCut<T>& operator<<(const EgHLTDQMCut<T>* inputCut);
    
    
    //basically an AND of all the cuts using short circuit evaluation, starting with the first cut
    //if no cuts present, will default to true
    bool pass(const T& obj,const OffEvt& evt)const;
    EgHLTDQMCut<T>* clone()const{return new EgMultiCut(*this);}
  };
  
  template<class T> EgMultiCut<T>::EgMultiCut(const EgMultiCut<T>& rhs)
    {
      for(size_t cutNr=0;cutNr<rhs.cuts_.size();cutNr++){
	cuts_.push_back(rhs.cuts_[cutNr]->clone());
      }
    }
  
  
  template<class T> EgMultiCut<T>& EgMultiCut<T>::operator<<(const EgHLTDQMCut<T>* inputCut)
    {
      if(typeid(*inputCut)==typeid(EgMultiCut)){
	edm::LogError("EgMultiCut") <<" Error can not currently load an EgMultiCut inside a EgMultiCut, the practical upshot is that the selection you think is being loaded isnt ";
      }else if(inputCut==NULL){
	edm::LogError("EgMultiCut") << "Error, cut being loaded is null, ignoring";
      }else cuts_.push_back(inputCut);
      return *this;
    }
  
  template<class T> bool EgMultiCut<T>::pass(const T& obj,const OffEvt& evt)const
    {
      for(size_t i=0;i<cuts_.size();i++){
	if(!cuts_[i]->pass(obj,evt)) return false;
      }
      return true;
    
    }
  
  //pass in which bits you want the trigger to pass
  //how this works
  //1) you specify the trigger bits you want to pass
  //2) you then specify whether you require all to be passed (AND) or just 1 (OR). It assumes OR by default
  //3) optionally, you can then specify any trigger bits you want to ensure fail. If any of these trigger bits 
  //   are passed (OR), then the cut fails, you can also specify only to fail if all are passed (AND)
  template<class T> struct EgObjTrigCut : public EgHLTDQMCut<T> {
    public:
    enum CutLogic{AND,OR};
    
    private:
    //currently fine for default copy construction
    TrigCodes::TrigBitSet bitsToPass_; 
    CutLogic passLogic_;
    TrigCodes::TrigBitSet bitsToFail_;
    CutLogic failLogic_;					
    
    public:
    EgObjTrigCut( TrigCodes::TrigBitSet bitsToPass,CutLogic passLogic=OR,TrigCodes::TrigBitSet bitsToFail=TrigCodes::TrigBitSet(),CutLogic failLogic=AND):
      bitsToPass_(bitsToPass),passLogic_(passLogic),bitsToFail_(bitsToFail),failLogic_(failLogic){}
    ~EgObjTrigCut(){}
    
    bool pass(const T& obj,const OffEvt& evt)const;
    EgHLTDQMCut<T>* clone()const{return new EgObjTrigCut(*this);}
  };
  
  template<class T> bool EgObjTrigCut<T>::pass(const T& obj,const OffEvt& evt)const
    {
      TrigCodes::TrigBitSet passMasked = bitsToPass_&obj.trigBits();
      TrigCodes::TrigBitSet failMasked = bitsToFail_&obj.trigBits();
    
      bool passResult = passLogic_==AND ? passMasked==bitsToPass_ : passMasked!=0x0;
      bool failResult = failLogic_==AND ? failMasked==bitsToFail_ : failMasked!=0x0;
      if(bitsToFail_==0x0) failResult=false; //ensuring it has no effect if bits not specified
      return passResult && !failResult;
    
    }
  
  
  
  //pass in which bits you want the trigger to pass
  //can either specify to pass all of the bits (AND) or any of the bits (OR)
  template<class T> struct EgEvtTrigCut : public EgHLTDQMCut<T> {
    public:
    enum CutLogic{AND,OR};
    private:
    //currently default copy constructor is fine
    TrigCodes::TrigBitSet bitsToPass_;
    CutLogic passLogic_;
    
    public:
    EgEvtTrigCut( TrigCodes::TrigBitSet bitsToPass,CutLogic passLogic=OR):bitsToPass_(bitsToPass),passLogic_(passLogic){}
    ~EgEvtTrigCut(){}
    
    bool pass(const T& obj,const OffEvt& evt)const; 
    EgHLTDQMCut<T>* clone()const{return new EgEvtTrigCut(*this);}
  };
  
  template<class T> bool EgEvtTrigCut<T>::pass(const T& obj,const OffEvt& evt)const
    {
      TrigCodes::TrigBitSet passMasked = bitsToPass_&evt.evtTrigBits();
      return passLogic_==AND ? passMasked==bitsToPass_ : passMasked!=0x0;
    }
  
  //nots the cut, ie makes it return false instead of true
  template<class T> struct EgNotCut : public EgHLTDQMCut<T> {
    private:
    EgHLTDQMCut<T>* cut_; //we own it
    
    public:
    EgNotCut(EgHLTDQMCut<T>* cut):cut_(cut){}
    EgNotCut(const EgNotCut<T>& rhs):cut_(rhs.cut_->clone()){}
    ~EgNotCut(){delete cut_;}
    
    bool pass(const T& obj,const OffEvt& evt)const{return !cut_->pass(obj,evt);}
    EgHLTDQMCut<T>* clone()const{return new EgNotCut(*this);}
  };
  
  //cut on the charge of the electron
  template<class T> struct ChargeCut : public EgHLTDQMCut<T> {
    private:
    int charge_;
    
    public:
    ChargeCut(int charge):charge_(charge){}
    ~ChargeCut(){}
    
    bool pass(const T& obj,const OffEvt& evt)const{return obj.charge()==charge_;}
    EgHLTDQMCut<T>* clone()const{return new ChargeCut(*this);}
  };
  
  //this askes if an object statifies the probe criteria and that another electron in the event statisfies the tag
  //although templated, its hard to think of this working for anything else other then an electron
  template<class T> struct EgTagProbeCut : public EgHLTDQMCut<T> {
    private:
    int probeCutCode_; 
    int (T::*probeCutCodeFunc_)()const;
    int tagCutCode_;
    int (OffEle::*tagCutCodeFunc_)()const;
    float minMass_;
    float maxMass_;
    public:
    EgTagProbeCut(int probeCutCode,int (T::*probeCutCodeFunc)()const,int tagCutCode,int (OffEle::*tagCutCodeFunc)()const,float minMass=81.,float maxMass=101.):probeCutCode_(probeCutCode),probeCutCodeFunc_(probeCutCodeFunc),tagCutCode_(tagCutCode),tagCutCodeFunc_(tagCutCodeFunc),minMass_(minMass),maxMass_(maxMass){}
    ~EgTagProbeCut(){}
    
    bool pass(const T& obj,const OffEvt& evt)const;
    EgHLTDQMCut<T>* clone()const{return new EgTagProbeCut(*this);}
  };
  
  template<class T> bool EgTagProbeCut<T>::pass(const T& obj,const OffEvt& evt)const
    {
      int nrTags=0;
      const OffEle* tagEle=NULL;
      const std::vector<OffEle>& eles = evt.eles();
      //we are looking for an *additional* tag
      for(size_t eleNr=0;eleNr<eles.size();eleNr++){
	if( ((eles[eleNr].*tagCutCodeFunc_)() & tagCutCode_)==0x0){
	  //now a check that the tag is not the same as the probe
	  if(reco::deltaR2(obj.eta(),obj.phi(),eles[eleNr].eta(),eles[eleNr].phi())>0.1*0.1){//not in a cone of 0.1 of probe object
	    nrTags++;
	    tagEle = &eles[eleNr];
	  }
	}
      }
      if(nrTags==1){ //we are requiring one and only one additional tag (the obj is automatically excluded from the tag list)
	if(((obj.*probeCutCodeFunc_)() & probeCutCode_)==0x0){ //passes probe requirements, lets check the mass
	  float mass = (obj.p4()+tagEle->p4()).mag();
	  if(mass>minMass_ && mass<maxMass_) return true; //mass requirements
	}
      }
      return false; 
    }
  
  template<class T> struct EgJetTagProbeCut : public EgHLTDQMCut<T>{
    private:
    int probeCutCode_;
    int (OffEle::*probeCutCodeFunc_)()const;

    float minDPhi_;
    float maxDPhi_;
    public:
    EgJetTagProbeCut(int probeCutCode,int (T::*probeCutCodeFunc)()const,float minDPhi=-M_PI,float maxDPhi=M_PI):
      probeCutCode_(probeCutCode),probeCutCodeFunc_(probeCutCodeFunc),minDPhi_(minDPhi),maxDPhi_(maxDPhi){}
    bool pass(const T& obj,const OffEvt& evt)const;
    EgHLTDQMCut<T>* clone()const{return new EgJetTagProbeCut(*this);}
    
  };
  
  
  template<class T> bool EgJetTagProbeCut<T>::pass(const T& obj,const OffEvt& evt)const
    {
      int nrProbes=0;
      const std::vector<OffEle>& eles = evt.eles();
      for(size_t eleNr=0;eleNr<eles.size();eleNr++){
	if( ((eles[eleNr].*probeCutCodeFunc_)() & probeCutCode_)==0x0){
	  nrProbes++;
	}
      }
      bool b2bJet=false;
      const std::vector<reco::CaloJet>& jets =evt.jets();
      for(size_t jetNr=0;jetNr<jets.size();jetNr++){
	if(reco::deltaR2(obj.eta(),obj.phi(),jets[jetNr].eta(),jets[jetNr].phi())>0.1*0.1){//not in a cone of 0.1 of probe object
	  float dPhi = reco::deltaPhi(obj.phi(),jets[jetNr].phi());
	  if(dPhi>minDPhi_ && dPhi<maxDPhi_) b2bJet=true;
	}
      }
    
      return nrProbes==1 && b2bJet;
    
    }
  
  
  template<class T> struct EgJetB2BCut : public EgHLTDQMCut<T>{
    private:
    
    float minDPhi_;
    float maxDPhi_;
    float ptRelDiff_;
    
    public:
    EgJetB2BCut(float minDPhi=-M_PI,float maxDPhi=M_PI,float ptRelDiff=999):
      minDPhi_(minDPhi),maxDPhi_(maxDPhi),ptRelDiff_(ptRelDiff){}
    bool pass(const T& obj,const OffEvt& evt)const;
    EgHLTDQMCut<T>* clone()const{return new EgJetB2BCut(*this);}
    
  };
  

  template<class T> bool EgJetB2BCut<T>::pass(const T& obj,const OffEvt& evt)const
    {
    
      bool b2bJet=false;
      const std::vector<reco::CaloJet>& jets =evt.jets();
      for(size_t jetNr=0;jetNr<jets.size();jetNr++){
	if(reco::deltaR2(obj.eta(),obj.phi(),jets[jetNr].eta(),jets[jetNr].phi())>0.1*0.1){//not in a cone of 0.1 of probe object
	  float dPhi = reco::deltaPhi(obj.phi(),jets[jetNr].phi());
	  if(dPhi>minDPhi_ && dPhi<maxDPhi_ && fabs(1-jets[jetNr].pt()/obj.pt()) < ptRelDiff_) b2bJet=true;
	}
      }
      return b2bJet;
    
    }
  
  
  //requires the the passed in electron and another in the event passes the specified cuts
  struct EgDiEleCut : public EgHLTDQMCut<OffEle> {
    private:
    int cutCode_;
    int (OffEle::*cutCodeFunc_)()const;
    
    public:
    EgDiEleCut(int cutCode,int (OffEle::*cutCodeFunc)()const):cutCode_(cutCode),cutCodeFunc_(cutCodeFunc){}
    bool pass(const OffEle& obj,const OffEvt& evt)const;
    EgHLTDQMCut<OffEle>* clone()const{return new EgDiEleCut(*this);}
  };
  
  //requires the the passed in electron and another in the event passes the specified cuts
  template<class Key> struct EgDiEleUserCut : public EgHLTDQMCut<OffEle> {
    private:
    int (OffEle::*cutCodeFunc_)(const Key&)const;   
    const Key& key_;
    int cutsNotToMask_;
    public:
    EgDiEleUserCut(int (OffEle::*cutCodeFunc)(const Key&)const,const Key& key,int cutsNotToMask=~0x0):cutCodeFunc_(cutCodeFunc),key_(key),cutsNotToMask_(cutsNotToMask){}
    ~EgDiEleUserCut(){}
    
    bool pass(const OffEle& obj,const OffEvt& evt)const;
    EgHLTDQMCut<OffEle>* clone()const{return new EgDiEleUserCut(*this);} //default copy constructor is fine
    
  };
  
  template<class Key> bool EgDiEleUserCut<Key>::pass(const OffEle& obj,const OffEvt& evt)const
    { 
      const std::vector<OffEle>& eles = evt.eles();
      for(size_t eleNr=0;eleNr<eles.size();eleNr++){
	if(&eles[eleNr]!=&obj){ //different electrons
	  int diEleCutCode = (obj.*cutCodeFunc_)(key_) | (eles[eleNr].*cutCodeFunc_)(key_);  
	  if( (diEleCutCode & cutsNotToMask_)==0x0) return true;
	}
      }
      return false;
    }

  
  //requires the the passed in photon and another in the event passes the specified cuts
  struct EgDiPhoCut : public EgHLTDQMCut<OffPho> {
    private:
    int cutCode_;
    int (OffPho::*cutCodeFunc_)()const;
    
    public:
    EgDiPhoCut(int cutCode,int (OffPho::*cutCodeFunc)()const):cutCode_(cutCode),cutCodeFunc_(cutCodeFunc){}
    bool pass(const OffPho& obj,const OffEvt& evt)const;
    EgHLTDQMCut<OffPho>* clone()const{return new EgDiPhoCut(*this);}
  };
  
  
  //requires passed photon and another in the event passes the specified cuts
  template<class Key> struct EgDiPhoUserCut : public EgHLTDQMCut<OffPho> {
    private:
    int (OffPho::*cutCodeFunc_)(const Key&)const;   
    const Key& key_;
    int cutsNotToMask_;
    public:
    EgDiPhoUserCut(int (OffPho::*cutCodeFunc)(const Key&)const,const Key& key,int cutsNotToMask=~0x0):cutCodeFunc_(cutCodeFunc),key_(key),cutsNotToMask_(cutsNotToMask){}
    ~EgDiPhoUserCut(){}
    
    bool pass(const OffPho& obj,const OffEvt& evt)const;
    EgHLTDQMCut<OffPho>* clone()const{return new EgDiPhoUserCut(*this);} //default copy constructor is fine
    
  };
  
  template<class Key> bool EgDiPhoUserCut<Key>::pass(const OffPho& obj,const OffEvt& evt)const
    { 
      const std::vector<OffPho>& phos = evt.phos();
      for(size_t phoNr=0;phoNr<phos.size();phoNr++){
	if(&phos[phoNr]!=&obj){ //different phoctrons
	
	  int diPhoCutCode = (obj.*cutCodeFunc_)(key_) | (phos[phoNr].*cutCodeFunc_)(key_);
	  if( (diPhoCutCode & cutsNotToMask_)==0x0) return true;
	}
      }
      return false;
    }
  
  //a trigger tag and probe cut
  //basically we require the electron to pass some cuts
  //and then do tag and probe on the trigger
  //removing templates as it makes no sense
  struct EgTrigTagProbeCut : public EgHLTDQMCut<OffEle> {
    private:
    TrigCodes::TrigBitSet bitsToPass_;
    int cutCode_;
    int (OffEle::*cutCodeFunc_)()const;
    float minMass_;
    float maxMass_;
    public:
    EgTrigTagProbeCut(TrigCodes::TrigBitSet bitsToPass,int cutCode,int (OffEle::*cutCodeFunc)()const,float minMass=81.,float maxMass=101.):bitsToPass_(bitsToPass),cutCode_(cutCode),cutCodeFunc_(cutCodeFunc),minMass_(minMass),maxMass_(maxMass){}
    ~EgTrigTagProbeCut(){}
    
    bool pass(const OffEle& ele,const OffEvt& evt)const;
    EgHLTDQMCut<OffEle>* clone()const{return new EgTrigTagProbeCut(*this);} 
    
  };
  
  //----Morse----
  //new tag and probe cut
  //require two wp80 electrons
  struct EgTrigTagProbeCut_New : public EgHLTDQMCut<OffEle> {
    private:
    TrigCodes::TrigBitSet bit1ToPass_;
    TrigCodes::TrigBitSet bit2ToPass_;
    int cutCode_;
    int (OffEle::*cutCodeFunc_)()const;
    float minMass_;
    float maxMass_;
    public:
    EgTrigTagProbeCut_New(TrigCodes::TrigBitSet bit1ToPass,TrigCodes::TrigBitSet bit2ToPass,int cutCode,int (OffEle::*cutCodeFunc)()const,float minMass=81.,float maxMass=101.):bit1ToPass_(bit1ToPass),bit2ToPass_(bit2ToPass),cutCode_(cutCode),cutCodeFunc_(cutCodeFunc),minMass_(minMass),maxMass_(maxMass){}
    ~EgTrigTagProbeCut_New(){}
    
    bool pass(const OffEle& ele,const OffEvt& evt)const;
    EgHLTDQMCut<OffEle>* clone()const{return new EgTrigTagProbeCut_New(*this);} 
    
  };
  //same for photons
  struct EgTrigTagProbeCut_NewPho : public EgHLTDQMCut<OffPho> {
    private:
    TrigCodes::TrigBitSet bit1ToPass_;
    TrigCodes::TrigBitSet bit2ToPass_;
    int cutCode_;
    int (OffPho::*cutCodeFunc_)()const;
    float minMass_;
    float maxMass_;
    public:
    EgTrigTagProbeCut_NewPho(TrigCodes::TrigBitSet bit1ToPass,TrigCodes::TrigBitSet bit2ToPass,int cutCode,int (OffPho::*cutCodeFunc)()const,float minMass=81.,float maxMass=101.):bit1ToPass_(bit1ToPass),bit2ToPass_(bit2ToPass),cutCode_(cutCode),cutCodeFunc_(cutCodeFunc),minMass_(minMass),maxMass_(maxMass){}
    ~EgTrigTagProbeCut_NewPho(){}
    
    bool pass(const OffPho& pho,const OffEvt& evt)const;
    EgHLTDQMCut<OffPho>* clone()const{return new EgTrigTagProbeCut_NewPho(*this);} 
    
  };

}//end of namespace
#endif
