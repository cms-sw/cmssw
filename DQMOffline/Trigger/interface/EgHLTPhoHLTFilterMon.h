#ifndef DQMOFFLINE_TRIGGER_EGHLTPHOHLTFILTERMON
#define DQMOFFLINE_TRIGGER_EGHLTPHOHLTFILTERMON

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


#include "DQMOffline/Trigger/interface/EgHLTMonElemContainer.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemManager.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemWithCut.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemFuncs.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEvt.h"
#include "DQMOffline/Trigger/interface/EgHLTParticlePair.h"
#include "DQMOffline/Trigger/interface/EgHLTOffPho.h"
#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"

#include <string>

namespace trigger{
  class TriggerObject;
}

namespace egHLT {
  struct BinData;
  struct CutMasks;

  class PhoHLTFilterMon {
    
  public:
    
    
    //comparision functor for PhoHLTFilterMon
    //short for: pointer compared with string
    struct ptrCompStr : public std::binary_function<const PhoHLTFilterMon*,const std::string&,bool> {
      bool operator()(const PhoHLTFilterMon* lhs,const std::string& rhs){return lhs->filterName()<rhs;}
      bool operator()(const std::string& lhs,const PhoHLTFilterMon* rhs){return lhs<rhs->filterName();}
    };
    //yes I am aware that such a function undoubtably exists but it was quicker to write it than look it up
    template<class T> struct ptrLess : public std::binary_function<const T*,const T*,bool> {
      bool operator()(const T* lhs,const T* rhs){return *lhs<*rhs;}
    };
    
  private:
    std::string filterName_;
    const TrigCodes::TrigBitSet filterBit_;
    
    //we own the pointers in the vectors
    std::vector<MonElemManagerBase<trigger::TriggerObject>*> trigMonElems_;
    std::vector<MonElemContainer<OffPho>*> phoEffHists_;
    std::vector<MonElemContainer<OffPho>*> phoMonElems_;
    std::vector<MonElemContainer<OffPho>*> phoFailMonElems_;
    
    //we also own these pointers
    MonElemManagerBase<ParticlePair<OffPho> >* diPhoMassBothME_;
    MonElemManagerBase<ParticlePair<OffPho> >* diPhoMassOnlyOneME_; 
    MonElemManagerBase<ParticlePair<OffPho> >* diPhoMassBothHighME_;
    MonElemManagerBase<ParticlePair<OffPho> >* diPhoMassOnlyOneHighME_;

    //disabling copying
    PhoHLTFilterMon(const PhoHLTFilterMon&){}
    PhoHLTFilterMon& operator=(const PhoHLTFilterMon&){return *this;}
  public:
    PhoHLTFilterMon(MonElemFuncs& monElemFuncs, const std::string& filterName,TrigCodes::TrigBitSet filterBit,const BinData& bins,const CutMasks& masks);
    ~PhoHLTFilterMon();
    
    
    void fill(const OffEvt& evt,float weight);
    
    //sort by filter name
    bool operator<(const PhoHLTFilterMon& rhs)const{return filterName_<rhs.filterName_;}
    //bool operator<(const std::string& rhs)const{return filterName_<rhs;}
    const std::string& filterName()const{return filterName_;}
    
  };
}



#endif
