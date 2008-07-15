#ifndef DQMOFFLINE_TRIGGER_ELEHLTFILTERMON
#define DQMOFFLINE_TRIGGER_ELEHLTFILTERMON

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
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DQMOffline/Trigger/interface/MonElemManager.h"
#include "DQMOffline/Trigger/interface/EgHLTOffData.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"
#include "DQMOffline/Trigger/interface/EgammaHLTEffSrc.h"

namespace trigger{
  class TriggerObject;
}

class EleHLTFilterMon {

 public:
  //a helper struct to store the object collection handles to faciliate passing them in functions
 
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
  //we own the pointers in the vectors
  std::vector<MonElemManagerBase<EgHLTOffEle>*> eleMonElems_;
  std::vector<MonElemManagerBase<trigger::TriggerObject>*> trigMonElems_;
  std::vector<MonElemManagerBase<EgHLTOffEle>*> eleFailMonElems_;
  std::vector<EgammaHLTEffSrcBase<EgHLTOffEle>*> eleEffHists_;
  //std::vector<MonElemManager*> l1EleMonElems_;
  std::string filterName_;
  
  
  //disabling copying
  EleHLTFilterMon(const EleHLTFilterMon&){}
  EleHLTFilterMon& operator=(const EleHLTFilterMon&){return *this;}
 public:
  explicit EleHLTFilterMon(std::string filterName);
  ~EleHLTFilterMon();
  
  //should really be in a helper class...
  static void initStdEleHists(std::vector<MonElemManagerBase<EgHLTOffEle>*>& histVec,std::string baseName); //passing by value intentionally
  static void initStdEffHists(std::vector<EgammaHLTEffSrcBase<EgHLTOffEle>*>& histVec,std::string baseName,int nrBins,double xMin,double xMax,float (EgHLTOffEle::*vsVarFunc)()const);

  void fill(const EgHLTOffData& evtData,float weight);
  
  //sort by filter name
  bool operator<(const EleHLTFilterMon& rhs)const{return filterName_<rhs.filterName_;}
  //bool operator<(const std::string& rhs)const{return filterName_<rhs;}
  const std::string& filterName()const{return filterName_;}
  
};




#endif
