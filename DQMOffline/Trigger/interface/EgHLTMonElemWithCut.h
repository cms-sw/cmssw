#ifndef DQMOFFLINE_TRIGGER_EGHLTMONELEMWITHCUT
#define DQMOFFLINE_TRIGGER_EGHLTMONELEMWITHCUT


//class: MonElemWithCut, short for MonitorElementWithCut (note not MonEleWith Cut as Ele might be confused for electron
//
//author: Sam Harper (Aug 2008)
//
//WARNING: interface is NOT final, please dont use this class for now without clearing it with me
//         as I will change it and possibly break all your code
//
//aim:  to improve the fire and forget nature of MonElemManger
//      allows some arbitary selection to be placed on objects used to fill the monitoring element
//      examples include it having to be associated with a specific trigger filter

//implimentation: uses a MonElemManager to handle the Monitor Element and EgHLTDQMCut descide whether to fill it or not
//                it was debated adding this capacity directly to MonElemManager 
//                this may happen in a future itteration of the code when things have stabilised



#include "DQMOffline/Trigger/interface/EgHLTMonElemManager.h"
#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEvt.h"
namespace egHLT {
  template<class T> class MonElemWithCutBase {
    
  private:
    MonElemWithCutBase(const MonElemWithCutBase& rhs){}
    MonElemWithCutBase& operator=(const MonElemWithCutBase& rhs){return *this;}
  public:
    MonElemWithCutBase(){}
    virtual ~MonElemWithCutBase(){}
    
    virtual void fill(const T& obj,const OffEvt& evt ,float weight)=0;
    
  };
  
  
  template<class T,typename varTypeX,typename varTypeY=varTypeX> class MonElemWithCut : public MonElemWithCutBase<T> {
    
  private:
    MonElemManagerBase<T>* monElemMgr_; //we own this
    const EgHLTDQMCut<T>* cut_; //we also own this
    
  private:
    MonElemWithCut(const MonElemWithCut& rhs){}
    MonElemWithCut& operator=(const MonElemWithCut& rhs){return *this;}
  public:
    
    MonElemWithCut(DQMStore::IBooker &iBooker,const std::string& name,const std::string& title,int nrBins,double xMin,double xMax,
		   varTypeX (T::*varFunc)()const,const EgHLTDQMCut<T>* cut=NULL):
      monElemMgr_(new MonElemManager<T,varTypeX>(iBooker,name,title,nrBins,xMin,xMax,varFunc)),
      cut_(cut){}
    
    MonElemWithCut(DQMStore::IBooker &iBooker,const std::string& name,const std::string& title,int nrBinsX,double xMin,double xMax,int nrBinsY,double yMin,double yMax,
		   varTypeX (T::*varFuncX)()const,varTypeY (T::*varFuncY)()const,const EgHLTDQMCut<T>* cut=NULL):
      monElemMgr_(new MonElemManager2D<T,varTypeX,varTypeY>(iBooker,name,title,nrBinsX,xMin,xMax,nrBinsY,yMin,yMax,varFuncX,varFuncY)),
      cut_(cut){}
    ~MonElemWithCut();
    
    void fill(const T& obj,const OffEvt& evt,float weight);
    
  };
  
  template<class T,typename varTypeX,typename varTypeY> 
  MonElemWithCut<T,varTypeX,varTypeY>::~MonElemWithCut()
  {
    if(cut_) delete cut_;
    if(monElemMgr_) delete monElemMgr_;
  }
  
  template<class T,typename varTypeX,typename varTypeY> 
  void MonElemWithCut<T,varTypeX,varTypeY>::fill(const T& obj,const OffEvt& evt,float weight)
  {
    if(cut_==NULL || cut_->pass(obj,evt)) monElemMgr_->fill(obj,weight);
  }
  
}

#endif
