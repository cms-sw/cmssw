#ifndef DQMOFFLINE_TRIGGER_EGHLTMONELEMCONTAINER
#define DQMOFFLINE_TRIGGER_EGHLTMONELEMCONTAINER

//class: MonElemContainer, short for Monitor Element Container 

//author: Sam Harper (Aug 2008)
//
//WARNING: interface is NOT final, please dont use this class for now without clearing it with me
//         as I will change it and possibly break all your code
//
//aim:  to improve the fire and forget nature of MonElemManger
//      holds a collection on monitor elements for which there is a global cut
//      for example: they all have to pass a paricular trigger

//implimentation: a cut to pass and then a list of monitor elements to fill 
//                two seperate vectors of MonitorElements and MonElemsWithCuts

#include "DQMOffline/Trigger/interface/EgHLTMonElemManager.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemWithCut.h"
#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEvt.h"

#include <string>
#include <vector>
namespace egHLT {
  template<class T> class MonElemContainer : public MonElemWithCutBase<T> {
    
  private:
    std::string baseName_;
    std::string baseTitle_;
    
    //so I want the ability to have both normal monitor elements and monitor elements with indivdual cuts
    //so untill the two classes are merged, I just have two vectors
    std::vector<MonElemWithCutBase<T>*> cutMonElems_; //we own these
    std::vector<MonElemManagerBase<T>*> monElems_; //we own these
    EgHLTDQMCut<T>* cut_; //we also own this
    
    
    
  private:
    MonElemContainer(const MonElemContainer& rhs){}
    MonElemContainer& operator=(const MonElemContainer& rhs){return *this;}
  public:
    
    MonElemContainer(std::string baseName="",std::string baseTitle="",
		     EgHLTDQMCut<T>* cut=NULL):
      baseName_(baseName),
      baseTitle_(baseTitle),
      cut_(cut){}
    
    ~MonElemContainer();
    
  //yes this is little more than a struct with some unnecessary function wrapers
    std::vector<MonElemWithCutBase<T>*>& cutMonElems(){return cutMonElems_;}
    const std::vector<MonElemWithCutBase<T>*>& cutMonElems()const{return cutMonElems_;}
    std::vector<MonElemManagerBase<T>*>& monElems(){return monElems_;}
    const std::vector<MonElemManagerBase<T>*>& monElems()const{return monElems_;}
    
    
    const std::string& name()const{return baseName_;}
    const std::string& title()const{return baseTitle_;}
    
    void fill(const T& obj,const OffEvt& evt,float weight);
    
  };
  
  template<class T> MonElemContainer<T>::~MonElemContainer()
  {
    for(size_t i=0;i<monElems_.size();i++) delete monElems_[i];
    for(size_t i=0;i<cutMonElems_.size();i++) delete cutMonElems_[i];
    if(cut_!=NULL) delete cut_;
  }
  
  
  template<class T> void MonElemContainer<T>::fill(const T& obj,const OffEvt& evt,float weight)
  {
    if(cut_==NULL || cut_->pass(obj,evt)){
    for(size_t i=0;i<monElems_.size();i++) monElems_[i]->fill(obj,weight);
    for(size_t i=0;i<cutMonElems_.size();i++) cutMonElems_[i]->fill(obj,evt,weight);
    }
  }
}
#endif
