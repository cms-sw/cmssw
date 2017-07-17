#ifndef DQMOFFLINE_TRIGGER_EGHLTMONELEMMGREBEE
#define DQMOFFLINE_TRIGGER_EGHLTMONELEMMGREBEE

//struct: MonElemMgrEBEE (Monitor Element Manger Barrel and Endcap)
//
//author: Sam Harper (July 2008)
//
//WARNING: interface is NOT final, please dont use this class for now without clearing it with me
//         as I will change it and possibly break all your code
//
//aim:  a monitor element which seperates transparently the objects into barrel and endcap
//
//implimentation: class simply has two MonElemMangers, one for endcap electrons, one for barrel electrons
//                and fills them approprately. It assumes that the class passed in has a detEta function 
//                and uses 1.5 as the barrel,endcap descriminate
//       
//      

#include "DQMOffline/Trigger/interface/EgHLTMonElemManager.h"

namespace egHLT {
  template<class T,typename varType> class MonElemMgrEBEE : public MonElemManagerBase<T>{
  private:
    MonElemManager<T,varType> barrel_;
    MonElemManager<T,varType> endcap_;
    
  public: 
    MonElemMgrEBEE(DQMStore::IBooker &iBooker, const std::string& name,const std::string& title,int nrBins,float min,float max,varType (T::*varFunc)()const):
      barrel_(iBooker, name+"_eb","Barrel "+title,nrBins,min,max,varFunc),
      endcap_(iBooker, name+"_ee","Endcap "+title,nrBins,min,max,varFunc){}
    
    ~MonElemMgrEBEE(){}
    
    void fill(const T& obj,float weight);
    
  };
  
  template<class T,typename varType> void MonElemMgrEBEE<T,varType>::fill(const T& obj,float weight)
  {
    if(fabs(obj.detEta())<1.5) barrel_.fill(obj,weight);
    else endcap_.fill(obj,weight);
  }
  




  template<class T,typename varTypeX,typename varTypeY> class MonElemMgr2DEBEE : public MonElemManagerBase<T>{
    
  private:
    MonElemManager2D<T,varTypeX,varTypeY> barrel_;
    MonElemManager2D<T,varTypeX,varTypeY> endcap_;
    
  public:
    MonElemMgr2DEBEE(DQMStore::IBooker &iBooker, const std::string& name,const std::string& title,int nrBinsX,double xMin,double xMax,int nrBinsY,double yMin,double yMax,
		     varTypeX (T::*varFuncX)()const,varTypeY (T::*varFuncY)()const):
      barrel_(iBooker, name+"_eb","Barrel "+title,nrBinsX,xMin,xMax,nrBinsY,yMin,yMax,varFuncX,varFuncY),
      endcap_(iBooker, name+"_ee","Endcap "+title,nrBinsX,xMin,xMax,nrBinsY,yMin,yMax,varFuncX,varFuncY){}
    
    ~MonElemMgr2DEBEE(){}
    
    void fill(const T& obj,float weight);
    
  };
  
  template<class T,typename varTypeX,typename varTypeY> void MonElemMgr2DEBEE<T,varTypeX,varTypeY>::fill(const T& obj,float weight)
  {
    if(fabs(obj.detEta())<1.5) barrel_.fill(obj,weight);
    else endcap_.fill(obj,weight);
  }
}
#endif
  
