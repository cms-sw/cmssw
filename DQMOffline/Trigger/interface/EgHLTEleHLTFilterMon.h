#ifndef DQMOFFLINE_TRIGGER_EGHLTELEHLTFILTERMON
#define DQMOFFLINE_TRIGGER_EGHLTELEHLTFILTERMON

//class: EleHLTFilterMon
//
//author: Sam Harper (June 2008)
//
//WARNING: interface is NOT final, please dont use this class for now without clearing it with me
//         as I will change it and possibly break all your code
//
//aim: this object will manage all histograms associated with a particular HLT filter
//     ie histograms for computing efficiency of each filter step, id efficiency of gsf electrons passing trigger etc   
//
//implimentation: 
//       
//       

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DQMOffline/Trigger/interface/EgHLTMonElemContainer.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemManager.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemWithCut.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemFuncs.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEvt.h"
#include "DQMOffline/Trigger/interface/EgHLTParticlePair.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"
#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"

#include <string>

namespace trigger{
  class TriggerObject;
}

namespace egHLT {
  struct BinData;
  struct CutMasks;
  class EleHLTFilterMon {
    
  public:
    
    
    //comparision functor for EleHLTFilterMon
    //short for: pointer compared with string
    struct ptrCompStr : public std::binary_function<const EleHLTFilterMon*,const std::string&,bool> {
      bool operator()(const EleHLTFilterMon* lhs,const std::string& rhs){return lhs->filterName()<rhs;}
      bool operator()(const std::string& lhs,const EleHLTFilterMon* rhs){return lhs<rhs->filterName();}
    };
    //yes I am aware that such a function undoubtably exists but it was quicker to write it than look it up
    template<class T> struct ptrLess : public std::binary_function<const T*,const T*,bool> {
      bool operator()(const T* lhs,const T* rhs){return *lhs<*rhs;}
    };
    
  private:
    std::string filterName_;
    const TrigCodes::TrigBitSet filterBit_;
    
    //we own the pointers in the vectors
    //std::vector<MonElemManagerBase<OffEle>*> eleMonElems_;
    std::vector<MonElemManagerBase<trigger::TriggerObject>*> trigMonElems_;
    //  std::vector<MonElemManagerBase<OffEle>*> eleFailMonElems_;
    //  std::vector<MonElemWithCutBase<OffEle>*> eleEffHists_;  
    std::vector<MonElemContainer<OffEle>*> eleEffHists_;
    std::vector<MonElemContainer<OffEle>*> eleMonElems_;
    std::vector<MonElemContainer<OffEle>*> eleFailMonElems_;
    
    //we also own these pointers
    MonElemManagerBase<ParticlePair<OffEle> >* diEleMassBothME_;
    MonElemManagerBase<ParticlePair<OffEle> >* diEleMassOnlyOneME_;
    MonElemManagerBase<ParticlePair<OffEle> >* diEleMassBothHighME_;
    MonElemManagerBase<ParticlePair<OffEle> >* diEleMassOnlyOneHighME_; 
    
    //disabling copying
    EleHLTFilterMon(const EleHLTFilterMon&){}
    EleHLTFilterMon& operator=(const EleHLTFilterMon&){return *this;}
  public:
    EleHLTFilterMon(MonElemFuncs& monElemFuncs, const std::string& filterName,TrigCodes::TrigBitSet filterBit,const BinData& bins,const CutMasks& masks);
    ~EleHLTFilterMon();
    
    
    void fill(const OffEvt& evt,float weight);
    
    //sort by filter name
    bool operator<(const EleHLTFilterMon& rhs)const{return filterName_<rhs.filterName_;}
    //bool operator<(const std::string& rhs)const{return filterName_<rhs;}
    const std::string& filterName()const{return filterName_;}
    
  };
}



#endif
