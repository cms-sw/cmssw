#ifndef DQMOFFLINE_TRIGGER_EGHLTOFFEVT
#define DQMOFFLINE_TRIGGER_EGHLTOFFEVT

//struct: EgHLTOffData (Egamma HLT Offline Data)
//
//author: Sam Harper (July 2008)
//
//WARNING: interface is NOT final, please dont use this class for now without clearing it with me
//         as I will change it and possibly break all your code
//
//aim: this is a simple struct which allows all the data needed by the egamma offline HLT DQM  code to be passed in as single object
//     this includes the TriggerEvent handle and the vector of EgHLTOffEle at the moment
//
//implimentation: 
//       
//      

#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"
#include "DQMOffline/Trigger/interface/EgHLTOffPho.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"


namespace egHLT {
  //we own nothing....
  class OffEvt {
  private:
    //edm::Handle<trigger::TriggerEvent> trigEvt_;
    edm::Handle<std::vector<reco::CaloJet> > jets_;
    
    std::vector<OffEle> eles_;//egHLT::OffEle is lightweightish and handles copying okay hence it isnt a vector of pointers
    std::vector<OffPho> phos_;//egHLT::OffPho is lightweightish and handles copying okay hence it isnt a vector of pointers
    //const std::vector<reco::CaloJet>* jets_;
    
    TrigCodes::TrigBitSet evtTrigBits_; //the triggers that fired in the event all in a handy bit set

    
  public:
    OffEvt(){}
    ~OffEvt(){}
    
    //accessors
    //    const trigger::TriggerEvent& trigEvt()const{return *trigEvt_.product();}
    const std::vector<OffEle>& eles()const{return eles_;}
    std::vector<OffEle>& eles(){return eles_;}
    const std::vector<OffPho>& phos()const{return phos_;}
    std::vector<OffPho>& phos(){return phos_;}
    TrigCodes::TrigBitSet evtTrigBits()const{return evtTrigBits_;}
    const std::vector<reco::CaloJet>& jets()const{return *jets_.product();}

    //modifiers
    void clear();
    void setEvtTrigBits(TrigCodes::TrigBitSet bits){evtTrigBits_=bits;}
    void setJets(edm::Handle<std::vector<reco::CaloJet> > jets){jets_=jets;}

  };
}


#endif
